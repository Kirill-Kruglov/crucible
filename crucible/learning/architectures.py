"""
Network builders: PatternFilter (Binary Filter) and generic layer list -> nn.Module.
"""
import torch
import torch.nn as nn
from typing import List


class PatternFilter(nn.Module):
    """
    Binary Filter: for each candidate pattern from v0 -> include/exclude (Bernoulli).
    v1.4.1: default input_dim=192 (FEATURE_DIM from vectorizer).
    """
    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 128,
        max_patterns: int = 20,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pattern_scorer = nn.Linear(hidden_dim, max_patterns)
        self._max_patterns = max_patterns

    def forward(self, episode_features: torch.Tensor) -> torch.Tensor:
        h = self.encoder(episode_features)
        return self.pattern_scorer(h)

    def sample(self, episode_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(episode_features)
        probs = torch.sigmoid(logits)
        mask = torch.bernoulli(probs)
        log_prob = (
            mask * torch.log(probs + 1e-8)
            + (1 - mask) * torch.log(1 - probs + 1e-8)
        ).sum()
        return mask, log_prob


def build_from_layers(layers: List[dict], input_dim: int) -> nn.Module:
    """Build nn.Sequential from list of layer dicts. For binary_filter use PatternFilter directly."""
    modules = []
    in_dim = input_dim
    for L in layers:
        t = L.get("type", "")
        if t == "linear":
            out = L.get("out", L.get("out_dim", 32))
            modules.append(nn.Linear(in_dim, out))
            in_dim = out
        elif t == "relu":
            modules.append(nn.ReLU())
        elif t == "tanh":
            modules.append(nn.Tanh())
    return nn.Sequential(*modules) if modules else nn.Identity()
