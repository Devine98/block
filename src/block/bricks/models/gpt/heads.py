from typing import Dict, Optional, Tuple

import torch
from torch import nn

from block.bricks.models.modeling_utils import PreTrainedModel

from .gpt import GPT2


class GPT2LMHeadModel(PreTrainedModel, nn.Module):
    """Modeling gpt2 lm head.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
    ) -> None:
        PreTrainedModel.__init__(self)
        nn.Module.__init__(self)

        if config is None:
            config = {}
        self.config = config

        self.model = GPT2(self.config)
        self.lm_head = nn.Linear(
            self.config.get("n_embd", 768),
            self.config.get("vocab_size", 50257),
            bias=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.model(inputs, attention_mask=attention_mask)
        lm_logits = self.lm_head(x)
        loss = None
        if labels is not None:

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return loss, lm_logits