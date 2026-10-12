from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64  # Maximum number of tokens to generate (completion tokens only)
    ignore_eos: bool = False
    max_model_length: int | None = None  # Maximum total sequence length (prompt + completion)

    def __post_init__(self):
        # temperature == 0 selects greedy (argmax) decoding
        assert self.temperature == 0 or self.temperature >= 1e-6, "temperature must be 0 (greedy) or >= 1e-6"