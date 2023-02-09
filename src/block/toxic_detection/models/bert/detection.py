import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from block.bricks.tokenizations.bert.tokenization import Tokenizer as BertTokenizer
from block.toxic_detection.modeling_utils import PreTrainedModel
from block.toxic_detection.models.bert.classification_model import Classifier
from block.toxic_detection.models.bert.dirty_words_utils import (
    convert_to_unicode,
    is_whitespace_or_punctuation,
)
from block.utils.trie import Trie


class Detector(PreTrainedModel):
    """Toxic detector .

    Depending on `bert_tokenizer` and `bert_51m`.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = {}
        self.config = config

        self._tokenizer = BertTokenizer(maxlen=self._maxlen)
        self._classifier = Classifier(self.config)

        # tries without region
        self._trie = Trie()
        self._dirty_words = set()

        # tries with region
        self.region_trie = {}
        self.region_dirty_words = {}

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config
        self._vocab_size = config.get("vocab_size", 50000)
        self._hidden_size = config.get("hidden_size", 512)
        self._num_heads = config.get("num_heads", 8)
        self._maxlen = config.get("maxlen", 512)
        self._n_layers = config.get("n_layers", 8)
        self._tag_num = config.get("tag_num", 6)
        self._tags = config.get(
            "tags",
            [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ],
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def classifier(self):
        return self._classifier

    def _load(self, model: str) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (str):
                Model file need to be loaded.
                A string, the path of a pretrained model.

        Raises:
            ValueError: str model should be a path!
        """

        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_dir(model)
            elif os.path.isfile(model):
                dir = os.path.join(self._tmpdir.name, "toxic_detection_bert")
                if os.path.exists(dir):
                    pass
                else:
                    os.mkdir(dir)
                self._unzip2dir(model, dir)
                self._load_from_dir(dir)
            else:
                raise ValueError("""str model should be a path!""")

        else:
            raise ValueError("""str model should be a path!""")

    def _load_from_dir(self, model_dir: str) -> None:
        """Set model params from `model_file`.

        Args:
            model_dir (str):
                Dir containing model params.
        """
        model_files = os.listdir(model_dir)

        # config
        if "config.pkl" not in model_files:
            raise FileNotFoundError("""config should in model dir!""")

        config = self._load_pkl(os.path.join(model_dir, "config.pkl"))
        self.config = config

        # classifier
        if "classifier.pkl" not in model_files:
            raise FileNotFoundError("""classifier should in model dir!""")

        self._classifier = Classifier(self._config)
        self._classifier.load_state_dict(
            torch.load(os.path.join(model_dir, "classifier.pkl"), map_location="cpu")
        )
        self._classifier.eval()

        # dirty words
        if "dirty_words.pkl" not in model_files:
            raise FileNotFoundError("""dirty_words should in model dir!""")
        dirty_words = self._load_pkl(os.path.join(model_dir, "dirty_words.pkl"))

        if isinstance(dirty_words, Dict):
            for region in dirty_words:
                self.add_sensitive_words(dirty_words[region], region)
        else:
            self.add_sensitive_words(dirty_words)

        # bert_tokenizer
        if "bert_tokenizer.pkl" in model_files:
            self._tokenizer.from_pretrained(
                os.path.join(model_dir, "bert_tokenizer.pkl")
            )
        else:
            self._tokenizer.from_pretrained("bert_tokenizer")

    def add_sensitive_words(
        self, words: Iterable[str], region: Optional[str] = None
    ) -> None:
        """add dirty words set and build trie"""
        # add to dirty words set
        words = [convert_to_unicode(w).strip().lower() for w in words]
        words = [w for w in words if len(w) >= 1]
        words = set(words)

        if region is None:
            self._dirty_words.update(words)
        else:
            if region not in self.region_dirty_words:
                self.region_dirty_words[region] = set()
            self.region_dirty_words[region].update(words)

        # remove * from *word or word*
        dirty_words = []
        for w in words:
            if w[-1] == "*":
                w = w[:-1]
            if w[0] == "*":
                w = w[1:]
            if len(w) >= 1:
                dirty_words.append(w)

        # build trie from  words without *
        dirty_words = list(set(dirty_words))

        if region is None:
            for word in dirty_words:
                self._trie.add(word)
        else:
            if region not in self.region_trie:
                self.region_trie[region] = Trie()
            for word in dirty_words:
                self.region_trie[region].add(word)

    def _check_dirty_words_rules(
        self, text: str, word: str, start: int, end: int, region: Optional[str] = None
    ) -> bool:
        """checks whether `word` in blacklist."""
        right_punc = False
        left_punc = False

        if end >= len(text):
            right_punc = True
        elif is_whitespace_or_punctuation(text[end]):
            right_punc = True

        if start == 0:
            left_punc = True
        elif is_whitespace_or_punctuation(text[start - 1]):
            left_punc = True

        if region is None:
            dirty_words = self._dirty_words
        else:
            dirty_words = self.region_dirty_words[region]

        # only `word` in dirtywords
        if f"{word}*" not in dirty_words:
            if f"*{word}" not in dirty_words:
                return word

        # both `word*` and `*word` in dirtywords
        if f"{word}*" in dirty_words:
            if f"*{word}" in dirty_words:
                if left_punc and right_punc:
                    return f"*{word}*"

        # `word*` in `*word` not
        if f"{word}*" in dirty_words:
            if f"*{word}" not in dirty_words:
                if right_punc:
                    return f"{word}*"

        # `word*` not in  `*word` in
        if f"{word}*" not in dirty_words:
            if f"*{word}" in dirty_words:
                if left_punc:
                    return f"*{word}"

        return ""

    def sensitive_words_detect(
        self, text: str, region: Optional[str] = None
    ) -> List[Tuple[str, int, int]]:
        """sensitive_words_detect fot text (Multimode matching).

        Args:
            text (str):
                The text need to be multimode match.
            region (Optional[str]):
                region of text .
                defaults to None .

        Returns:
            List[Tuple[str,int,int]]:
                Pattern as (match token,start,end).
                Where `match token` = `text`[`start`:`end`]
        """
        text = convert_to_unicode(text)
        text = text.lower()
        if region is None:
            raw_res = self._trie.multimode_match(text)
        else:
            raw_res = self.region_trie[region].multimode_match(text)
        res = []
        while raw_res:
            word, start, end = raw_res.pop()
            rule = self._check_dirty_words_rules(text, word, start, end, region)
            if rule:
                res.append((word, start, end, rule))
        return res

    def score(self, text: str) -> Dict[str, float]:
        """Scoring the input text.

        Args:
            input (str):
                Text input.

        Returns:
            Dict[str,float]:
                The toxic score of the input .
        """
        text = convert_to_unicode(text)
        input = self._tokenizer.encode_tensor(
            text, maxlen=self.config.get("maxlen", 512)
        ).view(1, -1)
        toxic_score = self._classifier.score(input).view(-1).tolist()
        toxic_score = [round(s, 2) for s in toxic_score]
        res = dict(
            zip(
                self._tags,
                toxic_score,
            )
        )

        return res

    def batch_score(self, texts: List[str]) -> List[Dict[str, float]]:
        """Scoring the input text.

        Args:
            input (List[str]):
                Text input.

        Returns:
            List[Dict[str, float]]:
                The toxic score of the input .
        """
        texts = [convert_to_unicode(text) for text in texts]

        input = [
            self._tokenizer.encode_tensor(
                text, maxlen=self.config.get("maxlen", 512)
            ).view(1, -1)
            for text in texts
        ]
        input = torch.cat(input, dim=0)
        toxic_score = self._classifier.score(input).tolist()
        res = [
            dict(
                zip(
                    self._tags,
                    [round(s, 2) for s in score],
                )
            )
            for score in toxic_score
        ]
        return res

    def detect(self, text: str, region: Optional[str] = None) -> Dict:
        """Detects toxic contents and sensitive words for `text`.

        Args:
            text (str):
                The text need to be detected.
            region (Optional[str]):
                region of text .
                defaults to None .

        Returns:
            Dict:
                Pattern as {
                        "sensitive_words" :  List[Tuple[str, int, int]],
                        "toxic_score " : Dict[str,float]
                    }.
        """
        out = {
            "sensitive_words": self.sensitive_words_detect(text, region),
            "toxic_score": self.score(text),
        }
        return out
