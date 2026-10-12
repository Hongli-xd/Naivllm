import torch 
import torch.nn as nn


class SamplerLayer(nn.Module):
    """
    A custom sampler layer that selects elements from the input tensor
    based on provided indices.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        # temperature == 0 means greedy decoding (argmax); required for
        # deterministic outputs and standard speculative-decoding verification
        greedy = temperature <= 1e-6
        safe_temperature = torch.where(greedy, torch.ones_like(temperature), temperature)
        argmax_tokens = logits.argmax(dim=-1)
        scaled_logits = logits / safe_temperature.unsqueeze(-1)
        probs = torch.softmax(scaled_logits, dim=-1)
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        sample_tokens = (probs / noise).argmax(dim=-1)
        return torch.where(greedy, argmax_tokens, sample_tokens)
