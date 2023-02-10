import torch


def pad_list(text, maxlen=1024, pad=0):
    x = [pad] * maxlen
    length = len(text)
    if length > 0:
        if length <= maxlen:
            x[-length:] = text
        else:
            x = text[-maxlen:]
    return x


class DataSet(torch.utils.data.Dataset):
    def __init__(self, corpus, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_positions = config.get("n_positions", 1024)
        self.vocab_size = config.get("vocab_size", 1024)
        self.corpus = corpus
        self.length = len(corpus)

    def __len__(self):
        return self.length

    def process(self, dialogue):

        # tokenize
        seg = [0]
        words = [self.tokenizer.cls_token]
        speaker = 1  # speaker
        for text in dialogue:
            text = self.tokenizer.tokenize(text)
            words.extend(text)
            words.append(self.tokenizer.sep_token)
            seg.extend([speaker] * (len(text) + 1))
            speaker = speaker % 2 + 1

        # convert_tokens_to_id
        ints = self.tokenizer.convert_tokens_to_id(words)

        # pad
        ints = pad_list(ints, maxlen=self.n_positions, pad=0)
        seg = pad_list(seg, maxlen=self.n_positions, pad=0)

        # tensor
        ints = torch.tensor(ints).long()
        seg = torch.tensor(seg).long()

        return ints, seg

    def __getitem__(self, idx):
        return self.process(self.corpus[idx])
