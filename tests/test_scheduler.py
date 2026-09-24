from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


def test_max_tokens_is_not_off_by_one():
    scheduler = Scheduler(1, 32, 8, 4, eos=99)
    sequence = Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=True))
    scheduler.add_sequence(sequence)
    scheduled, _ = scheduler.schedule()
    scheduler.postprocess(scheduled, [3])
    assert not sequence.is_finished

    scheduled, _ = scheduler.schedule()
    scheduler.postprocess(scheduled, [4])
    assert sequence.is_finished
    assert sequence.completion_token_ids == [3, 4]
