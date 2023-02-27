import random

import torch
from models.bert import tokenization

from .. import data_utils


class DataSet(torch.utils.data.IterableDataset):
    def __init__(
        self, x=None, y=None, maxlen=512, maxpred=64, min_count=3, tokenizer=None
    ):
        super().__init__()
        if not tokenizer:
            self.tokenizer = tokenization.Tokenizer()
        self.min_count = min_count
        self.maxlen = maxlen
        self.maxpred = maxpred
        self.vocab_len = len(self.tokenizer.vocab)
        self.x = x
        self.y = y
        self.not_mask_label = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id}

    def _generate_data(self, data):
        while True:
            #             random.shuffle(data)
            for item in data:
                yield item

    def gen_data(self, gen):
        r = random.randint(1, 15)
        res = [self.tokenizer.cls_token]
        for i in range(r):
            text = next(gen)
            tokens = self.tokenizer.tokenize(text)
            while len(tokens) < self.min_count:
                text = next(gen)
                tokens = self.tokenizer.tokenize(text)
            res.extend(tokens)
            if len(res) >= self.maxlen - 1:
                res = res[: self.maxlen - 1]
                res.append(self.tokenizer.sep_token)
                break
            else:
                res.append(self.tokenizer.sep_token)
        return res

    def mask_sentence(self, tokens):
        tokens = [self.tokenizer._convert_token_to_id(token) for token in tokens]
        sample_tokens = []
        for i, token in enumerate(tokens):
            if token not in self.not_mask_label:
                sample_tokens.append(i)
        n_pred = min(self.maxpred, max(1, int(len(sample_tokens) * 0.15)))
        cand_maked_pos = random.sample(sample_tokens, n_pred)

        mask_pos, mask_label = [], []
        for pos in cand_maked_pos:
            mask_pos.append(pos)
            mask_label.append(tokens[pos])
            r = random.random()
            # 80%  token is masked
            if r < 0.8:
                tokens[pos] = self.tokenizer.mask_token_id
            elif r < 0.9:
                tokens[pos] = random.randint(999, self.vocab_len - 1)
        return tokens, mask_pos, mask_label

    def pad_sentence(self, sent, maxlen):
        sent = data_utils.pad_list(sent, maxlen=maxlen, pad=0)
        sent = torch.tensor(sent).long()
        return sent

    def process(self):
        self.gen1 = self._generate_data(self.x)
        self.gen2 = self._generate_data(self.y)
        while True:
            r = random.random()
            if r < 0.5:
                data = self.gen_data(self.gen1)
            else:
                data = self.gen_data(self.gen2)
            tokens, mask_pos, mask_label = self.mask_sentence(data)
            tokens = self.pad_sentence(tokens, maxlen=self.maxlen)
            mask_pos = self.pad_sentence(mask_pos, maxlen=self.maxpred)
            mask_label = self.pad_sentence(mask_label, maxlen=self.maxpred)
            yield tokens, mask_pos, mask_label

    def __iter__(self):
        return self.process()
