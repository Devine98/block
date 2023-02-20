import collections
import copy
import itertools
import os
import pickle
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn, optim

from block.spelling_correction.models.mlp import counting_utils
from block.spelling_correction.models.mlp.correction import Corrector
from block.spelling_correction.models.mlp.correction_utils import (
    word2int_diff,
    word_cnts,
)
from block.spelling_correction.models.mlp.rank_block import RankModel
from block.spelling_correction.trainers.mlp.datasets import RankSet


class Trainer:
    """Trainer for mlp corrector.

    Attributes:
        model (Corrector, optional):
            The model need to be trained.
            If None,trainer will create a new model for training.
            Defaults to None.
        use_gpu (bool, optional):
            If True trainer will use gpu for training rank model if gpu is available.
            Defaults to False.
    """

    def __init__(
        self, model: Optional[Corrector] = None, use_gpu: Optional[bool] = False
    ) -> None:
        if model is None:
            model = Corrector()
        self.model = model
        self.use_gpu = use_gpu
        self._device = torch.device(
            "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        )

        self._model_file = {}

    def add_never_correct(self, never_correct: Iterable[str]) -> None:
        """Model will not correct the words in never_correct after tokenizing."""
        if "never_correct" not in self._model_file:
            self._model_file["never_correct"] = set()
        self._model_file["never_correct"].update(set(never_correct))

    def add_m2c(self, m2c: Dict[str, Iterable[str]]) -> None:
        """Add missspellings to correct words mapping if supported."""
        if "m2c" not in self._model_file:
            self._model_file["_m2c"] = {}
        if isinstance(m2c, Dict):
            self._model_file["_m2c"].update(m2c)
        self._model_file["_c_words"] = set(
            itertools.chain(*self._model_file["_m2c"].values())
        )

    def train(
        self,
        corpus: Iterable[str],
        words: List[List[str]],
        labels: List[List[str]],
        correct_token: str = "<correct>",
        batch_size: int = 64,
        max_step: int = 10000,
        num_workers: int = 1,
    ) -> None:
        """Training the corrector.

        Args:
            corpus (Iterable[str]):
                Clean corpus for statistics information.
            words (List[List[str]]):
                Words containing missspellings.
            labels (List[List[str]]):
                Labels for `words`.
                If word is correct,label is `correct_token`.
                If word is missspelled ,label is thr correct form of this word.
            correct_token (str, optional):
                Correct token for `labels`.
                Defaults to '<correct>'.
            batch_size (int, optional):
                Batch size for rank model.
                Defaults to 64.
            max_step (int, optional):
                Max training step for rank model.
                Defaults to 10000.
            num_workers (int, optional):
                Num workers for multi threads counting.
                Defaults to 1.
        """

        # tokenize
        if "never_correct" not in self._model_file:
            self._model_file["never_correct"] = {}

        # Optimization is required here
        # It takes hours for 10m lines of corpus.
        self.model.add_never_correct(self._model_file["never_correct"])
        tokenized_corpus = [self.model.tokenizer.tokenize(text) for text in corpus]

        # get corpus description
        self._get_corpus_des(tokenized_corpus, num_workers)

        # init model
        self.model.set_params(copy.deepcopy(self._model_file))
        self.model.init_model()

        # training rank model
        train_set = RankSet(
            model=self.model,
            words=words,
            labels=labels,
            correct_token=correct_token,
        )
        self._train_rank_model(
            self.model.rank_model,
            train_set,
            num_workers,
            batch_size,
            max_step,
        )

    def _get_corpus_des(
        self, tokenized_corpus: Iterable[Iterable[str]], num_workers: int = 1
    ) -> None:
        """Get word counts, word2int mapping,etc."""

        # cnts
        cnts = word_cnts(tokenized_corpus)

        # words
        words = set(cnts)

        # word2int
        word2int = {"<S>": 1, "<E>": 2}
        word2int.update(word2int_diff(word2int, words))

        # cooccurance
        cooccurance = counting_utils.get_cooccurance(
            word2int=word2int,
            corpus=tokenized_corpus,
            window_size=self.model.window_size,
            num_workers=num_workers,
        )

        # update `model_file`
        self._model_file["_cnts"] = cnts
        self._model_file["_words"] = words
        self._model_file["_word2int"] = word2int
        self._model_file["_cooccurance"] = cooccurance

    def _train_rank_model(
        self,
        rank_model: RankModel,
        train_set: RankSet,
        num_workers: int = 1,
        batch_size: int = 64,
        max_step: int = 10000,
    ) -> None:
        """Training rank model for mlp corrector."""
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        rank_model.train()
        rank_model.to(self._device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(rank_model.parameters())
        loss_info = collections.deque(maxlen=100)

        for step, data in enumerate(train_loader):
            data = [d.to(self._device) for d in data]
            out = rank_model(data[0])
            loss = criterion(out, data[1])
            loss.backward()
            loss_info.append(loss.cpu().item())
            nn.utils.clip_grad_norm_(rank_model.parameters(), 2)
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"step {step} , loss : {round(np.mean(loss_info),4)}")

            if step > max_step:
                break

    def save_model(self, path: str = "./model.pkl") -> None:
        """Save model state dicts."""

        # rank model
        self._model_file["rank_model"] = self.model.rank_model.state_dict()

        self._save_file(self._model_file, path)

    def _save_file(self, file: Dict, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        with open(path, "wb") as f:
            pickle.dump(file, f)
