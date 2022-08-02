from typing import Optional

import torch
from torch import nn

from sprite.bricks.models.bert.embedding import Embeddings
from sprite.bricks.models.bert.encoder import Encoder
from sprite.bricks.models.nn_utils import get_pad_mask


class BERT(nn.Module):

    """Modeling bert.

    Attributes:
        vocab_size (int):
            Vocab size.
        hidden_size (int, optional):
            The dim of hidden layer.
            Defaults to 512.
        num_heads (int, optional):
            The number of attention heads.
            Defaults to 8.
        n_layers (int, optional):
            The number of encoder layers.
            Defaults to 8.
        maxlen (int, optional):
            Max length of sequence.
            Defaults to 512.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        n_layers: int = 8,
        maxlen: int = 512,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.n_layers = n_layers

        self.embed = Embeddings(vocab_size, maxlen=maxlen, hidden_size=hidden_size)
        self.encoders = Encoder(
            d_model=self.hidden_size, num_heads=self.num_heads, n_layers=self.n_layers
        )

        # Can also have linear and tanh here
        # This implemention has no nsp block.
        self.apply(self._init_params)

    def _init_params(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self, inputs: torch.Tensor, segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of bert.

        Args:
            inputs (torch.Tensor):
                The index of words. shape:(b,l)
            segment_ids (Optional[torch.Tensor], optional):
                The index of segments.
                This arg is not usually used.
                Defaults to None.

        Returns:
            torch.Tensor:
                BERT result. shape:(b,l,d)
        """
        x = self.embed(inputs, segment_ids)
        attn_mask = get_pad_mask(inputs)
        x = self.encoders(x, attn_mask)
        # can have h_pooled hera: fc(x[:,0])
        return x
