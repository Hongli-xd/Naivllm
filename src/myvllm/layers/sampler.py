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
        scaled_logits = logits / temperature.unsqueeze(-1)
        probs = torch.softmax(scaled_logits, dim=-1)
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        sample_tokens = (probs / noise).argmax(dim=-1)
        return sample_tokens
