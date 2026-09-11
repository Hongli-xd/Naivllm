# /home/selom/cuda/MinivLLM/src/myvllm/engine/spec_llm_engine.py
import torch
from myvllm.engine.llm_engine import LLMEngine
from myvllm.spec_engine.spec_scheduler import SpecScheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams


class SpecLLMEngine(LLMEngine):
    def __init__(self, config: dict):
        # 初始化草稿模型和目标模型
        self.draft_config = config.copy()
        self.draft_config['model_name_or_path'] = config.get('draft_model', 'Qwen/Qwen3-0.6B')
        
        self.target_config = config.copy()
        self.target_config['model_name_or_path'] = config.get('target_model', 'Qwen/Qwen3-0.6B')
        
        # 初始化模型运行器
        self.draft_runner = ModelRunner(self.draft_config, rank=0)
        self.target_runner = ModelRunner(self.target_config, rank=0)
        
        # 初始化投机解码调度器
        self.spec_scheduler = SpecScheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256),
            num_speculative_tokens=config.get("num_speculative_tokens", 5)
        )
        
    def speculative_step(self) -> tuple[list[int], bool]:
        """执行一个投机解码步骤"""
        # 1. 调度草稿模型生成投机令牌
        scheduled_sequences = self.spec_scheduler.schedule_draft(
            self.spec_scheduler.running
        )
        
        if not scheduled_sequences:
            return [], False
            
        # 2. 草稿模型前向传播生成投机令牌
        draft_outputs = self.draft_runner.call("run_speculative", scheduled_sequences)
        draft_tokens = draft_outputs['tokens']
        draft_logits = draft_outputs['logits']
        
        # 3. 准备验证输入
        verify_sequences, draft_tokens, draft_logits = self.spec_scheduler.schedule_verification(
            scheduled_sequences, draft_tokens, draft_logits
        )
        
        # 4. 目标模型验证
        target_outputs = self.target_runner.call("run_verify", verify_sequences)
        target_logits = target_outputs['logits']
        
        # 5. 拒绝采样和后处理
        accepted_counts = self.spec_scheduler.postprocess_speculative(
            scheduled_sequences, draft_tokens, draft_logits, target_logits
        )
        
        return accepted_counts, True
