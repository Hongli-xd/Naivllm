import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256)
        )
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            # 事件驱动的进程协调机制
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        self.verbose = config.get("verbose", False)
        self._closed = False
        atexit.register(self.exit)


    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
            return [], 0, is_prefill
        # run the model
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        self.add_token_ids(self.tokenizer.encode(prompt), sampling_params)

    def add_token_ids(self, token_ids: list[int], sampling_params: SamplingParams) -> int:
        sequence = Sequence(token_ids=token_ids, sampling_params=sampling_params)
        self.scheduler.add_sequence(sequence)
        return sequence.seq_id

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def _generate(self, request_ids: list[int]) -> dict:
        generated_tokens = {}
        start_t = time.perf_counter()
        first_token_t = None
        prefill_tokens = 0
        decode_tokens = 0
        while not self.scheduler.is_finished():
            step_start = time.perf_counter()
            outputs, num_processed_tokens, is_prefill = self.step()
            step_end = time.perf_counter()
            running_time = step_end - step_start + 1e-10
            if is_prefill:
                prefill_tokens += num_processed_tokens
                if first_token_t is None:
                    first_token_t = step_end
            else:
                decode_tokens += num_processed_tokens
            if self.verbose:
                phase = "prefill" if is_prefill else "decode"
                print(
                    f"{phase}: {num_processed_tokens} tokens, "
                    f"{num_processed_tokens / running_time:.2f} tokens/s"
                )
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        end_t = time.perf_counter()
        ordered_tokens = [generated_tokens[seq_id] for seq_id in request_ids]
        return {
            "text": [self.tokenizer.decode(tokens) for tokens in ordered_tokens],
            "token_ids": ordered_tokens,
            "metrics": {
                "total_s": end_t - start_t,
                "ttft_s": (first_token_t or end_t) - start_t,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "output_tokens": sum(len(tokens) for tokens in ordered_tokens),
                "prefix_cache": self.scheduler.block_manager.cache_stats(),
            },
        }

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> dict:
        request_ids = [self.add_token_ids(self.tokenizer.encode(p), sampling_params) for p in prompts]
        return self._generate(request_ids)

    def generate_token_ids(
        self, prompts: list[list[int]], sampling_params: SamplingParams
    ) -> dict:
        """Generate from pre-tokenized prompts for exact, tokenizer-free benchmarks."""
        request_ids = [self.add_token_ids(ids, sampling_params) for ids in prompts]
        return self._generate(request_ids)

    def pin_prefix(self, token_ids: list[int]) -> int:
        return self.scheduler.block_manager.pin_prefix(token_ids)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()

