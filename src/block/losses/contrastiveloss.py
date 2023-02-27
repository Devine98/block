import torch
from torch import nn
from torch.nn.functional import cosine_similarity, pairwise_distance


class ContrastiveLoss(nn.Module):
    """Contrastive Loss function.

    Attributes:
        margin (float, optional):
            The margin for clamping loss.
            Defaults to 1.0.
        distance (str,optional):
            Distance for getting similarity.
            Defaults to "cosine".
        reduction (str,optional):
            Ways for reduction.
            Defaults to "mean".

    """

    def __init__(self, margin=1.0, distance="cosine", reduction: str = "mean") -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

        if distance not in ("cosine", "euclidean"):
            raise Exception("invalid distance")
        self.distance = distance
        if self.distance == "cosine":
            self._distance_func = lambda x1, x2: 1.0 - cosine_similarity(x1, x2)
        else:
            self._distance_func = lambda x1, x2: pairwise_distance(x1, x2)

        self.reduction = reduction

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward function of Contrastive.

        Args:
            x1 (torch.Tensor):
                The positive tensor. shape:(b,d)
            x2 (torch.Tensor):
                The negtive tensor. shape:(b,d)
            y (torch.Tensor):
                The label tensor. shape:(b,)

        Returns:
            torch.Tensor:
                Contrastive result. shape:(b,)
        """

        d = self._distance_func(x1, x2)
        loss = y * torch.pow(d, 2) + (1 - y) * torch.pow(
            torch.clamp(self.margin - d, min=0.0), 2
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
