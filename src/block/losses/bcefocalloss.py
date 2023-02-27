import torch
from torch import nn


class BCEFocalLoss(nn.Module):
    """BCEFocalLoss function.

    Attributes:
        alpha (float, optional):
            Larger alpha gives more weight to positive samples .
            Defaults to 0.5.
        gamma (float, optional):
            Larger gamma gives more weight to hard samples .
            Defaults to 2.0.
        reduction (str,optional):
            Ways for reduction.
            Defaults to "mean".

    """

    def __init__(
        self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"
    ) -> None:
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function of MarginLoss.

        Args:
            pred (torch.Tensor):
                The pred tensor result with no sigmoid. shape:(b,)
            target (torch.Tensor):
                The true label tensor. shape:(b,)

        Returns:
            torch.Tensor:
                BCEFocalLoss result. shape:(b,)
        """
        probs = torch.sigmoid(pred)
        pt = probs.clamp(min=0.0001, max=0.9999)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * target * torch.log(pt) - (
            1 - self.alpha
        ) * pt**self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
