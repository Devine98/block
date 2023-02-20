import itertools
import random
from typing import Iterator, List, Tuple

import numpy as np
import torch

from block.spelling_correction.models.mlp.correction import Corrector
from block.spelling_correction.models.mlp.rank_block import make_feature


class RankSet(torch.utils.data.IterableDataset):
    """IterableDataset for rank block of mlp corrector.

    Attributes:
        model(Corrector):
            Model for training.
        words (List[List[str]]):
            Words containing missspellings.
        labels (List[List[str]]):
            Labels for `words`.
            If word is correct,label is `correct_token`.
            If word is missspelled ,label is thr correct form of this word.
        correct_token (str, optional):
            Correct token for `labels`.
            Defaults to '<correct>'.


    Raises:
        ValueError:
            Length of `words` and `labels` should be equal!
    """

    def __init__(
        self,
        model: Corrector,
        words: List[List[str]],
        labels: List[List[str]],
        correct_token: str = "<correct>",
    ) -> None:

        super().__init__()
        self._model = model

        if not len(words) == len(labels):
            raise ValueError("""Length of `words` and `labels` should be equal!""")

        self._words = words
        self._labels = labels
        self._correct_token = correct_token
        self._x_pos = np.array([])
        self._x_neg = np.array([])

        self._init_dataset()
        if not self._x_pos.size > 0 or not self._x_neg.size > 0:
            raise ValueError("""Labels are not valid!""")

    def _init_dataset(self) -> None:
        """Makes features for all training words ,both correct words and wrong words."""

        for i in range(len(self._words)):
            tokens = ["<S>"] + self._words[i] + ["<E>"]
            text_labels = (
                [self._correct_token] + self._labels[i] + [self._correct_token]
            )
            for j, token in enumerate(tokens):

                if j == 0 or (j == len(tokens) - 1):
                    continue

                if (
                    (token in self._model.never_correct)
                    or (not token.isalpha())
                    or (text_labels[j] == self._correct_token)
                ):
                    continue

                correct_word = text_labels[j]
                left_words = tokens[max(i - self._model.window_size, 0) : i]
                right_words = tokens[i + 1 : i + self._model.window_size + 1][::-1]

                # pos features
                features_pos = make_feature(
                    aim_word=token,
                    sim_words=[correct_word],
                    m2c=self._model.m2c,
                    cnts=self._model.cnts,
                    left_words=left_words,
                    right_words=right_words,
                    word2int=self._model.word2int,
                    cooccurance_map=self._model.cooccurance,
                )
                if self._x_pos.size > 0:
                    self._x_pos = np.concatenate([self._x_pos, features_pos], axis=0)
                else:
                    self._x_pos = features_pos

                # neg features
                sim = self._model.make_candidates(token, 2)
                if not sim:
                    continue

                features_neg = make_feature(
                    aim_word=token,
                    sim_words=sim,
                    m2c=self._model.m2c,
                    cnts=self._model.cnts,
                    left_words=left_words,
                    right_words=right_words,
                    word2int=self._model.word2int,
                    cooccurance_map=self._model.cooccurance,
                )
                pos_idx = []
                for idx, sim_word in enumerate(sim):
                    if sim_word == correct_word:
                        pos_idx.append(idx)
                if pos_idx:
                    features_neg = np.delete(features_neg, pos_idx, axis=0)

                if self._x_neg.size > 0:
                    self._x_neg = np.concatenate([self._x_neg, features_neg], axis=0)
                else:
                    self._x_neg = features_neg

    def _process(self) -> Iterator[Tuple]:
        """Yields a training data for rank model."""
        pos_gen = itertools.cycle(self._x_pos)
        neg_gen = itertools.cycle(self._x_neg)

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        idx = 0
        while True:
            idx += 1
            if worker_info is None:
                pass
            else:
                if idx % num_workers != worker_id:
                    continue
            r = random.random()
            if r < 0.2:
                feature = next(pos_gen)
                label = 1
            else:
                feature = next(neg_gen)
                label = 0

            feature = torch.tensor(feature).float()
            label = torch.tensor(label).float()
            yield feature, label

    def __iter__(self) -> Iterator:
        """Yields a training data for rank model.

        Yields:
            Iterator:
                A feature (both center and context) and label of a word.
        """
        return self._process()
