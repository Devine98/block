import torch
from torch import nn


class MarginLoss(nn.Module):
    """Margin loss function.

    Attributes:
        margin (float, optional):
            The margin for clamping loss.
            Defaults to 1.0.
        reduction (str,optional):
            Ways for reduction.
            Defaults to "mean".

    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward function of MarginLoss.

        Args:
            x1 (torch.Tensor):
                The positive score tensor. shape:(b,)
            x2 (torch.Tensor):
                The negtive score tensor. shape:(b,)

        Returns:
            torch.Tensor:
                MarginLoss result. shape:(b,)
        """
        loss = torch.pow(torch.clamp(x2 - x1 + self.margin, min=0.0), 2)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
