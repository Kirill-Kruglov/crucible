"""
PolicyNetwork: generic MLP for routing, meta-routing, strategy selection (v1.6 S4).
Architecture: Linear(input_dim, 64) -> ReLU -> Linear(64, n_actions) -> logits; predict returns label by argmax.
"""
import torch
import torch.nn as nn
from typing import List


class PolicyNetwork(nn.Module):
    """Generic policy MLP: features -> logits over n_actions. Labels map indices to names."""

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        labels: List[str] | None = None,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.labels = labels or [str(i) for i in range(n_actions)]
        assert len(self.labels) == n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, features: list[float] | torch.Tensor) -> str:
        """Single prediction: features -> label string."""
        if isinstance(features, (list, tuple)):
            x = torch.tensor([features], dtype=torch.float32)
        else:
            x = features.unsqueeze(0) if features.dim() == 1 else features
        self.eval()
        with torch.no_grad():
            logits = self(x)
            idx = logits.argmax(dim=-1).item() if logits.shape[0] == 1 else logits.argmax(dim=-1)
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.labels[min(idx, len(self.labels) - 1)]
