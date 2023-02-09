from typing import Dict, List, Union

import nltk
import pandas as pd

from block.phrase_mining.modeling_utils import PreTrainedModel
from block.phrase_mining.models.rf_model.feature_extractor import (
    FeatureExtractor,
    _find_phrase_idx,
    _tokenize,
)
from block.phrase_mining.models.rf_model.phrase_selector import PhraseSelector


class FreqPhraseMiner(PreTrainedModel):
    """An implement of domain frequent phrase extraction model

    Attributes:
        min_length (int):
            The minimal length of grams to count.
            Defaults to 1.
        max_length (int):
            The maximal length of grams to count.
            Defaults to 5.
        most_common (int):
            Consider top most_common phrases of each gram.
            Defaults to 50.
        topn (int):
            Final results will return topn phrases.
            Defaults to 50.

    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 5,
        most_common: int = 50,
        topn: int = 50,
    ) -> None:
        super().__init__()

        # model parameters
        self._min_length = min_length
        self._max_length = max_length
        self._most_common = most_common
        self._topn = topn
        self._domain_phrase = []

        # model methods
        # self._feature_extractor = FeatureExtractor(doc=self._doc)

    def _load(self, model: Union[str, Dict]) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (Union[str, Dict]):
                Model file need to be loaded.
                Can be either:
                    - A string, the path of a pretrained model.
                    - A state dict containing model weights.

        Raises:
            ValueError: model file should be a dict.
        """

        if isinstance(model, str):
            model_file = self._load_pkl(model)
        else:
            model_file = model

        self._model_file = model_file
        self.set_params(model_file)
        self.init_model()

    def set_params(self, model_file: Dict) -> None:
        if "phrase_scoring_model" in model_file:
            if "sklearn" in str(type(model_file["phrase_scoring_model"])):
                self._phrase_scoring_model = model_file["phrase_scoring_model"]

        if "domain_phrase" in model_file:
            if isinstance(model_file["domain_phrase"], List):
                self._domain_phrase = model_file["domain_phrase"]

    def init_model(self) -> None:
        """
        init frequent phrase extraction model, including:
            1. init nltk stopwords

        """
        # download nltk
        nltk.download("stopwords", download_dir=self._TEMP_PATH)
        nltk.data.path.append(self._TEMP_PATH)

    def _score_phrase(self, doc: List[str]) -> pd.DataFrame:
        """calculate the feature of phrases,
        and use model to score phrases with features

        Args:
            doc (List[str]): document to process
            min_length (int): minimal length of grams to count
            max_length (int): maximal length of grams to count
            most_common (int): topn most common phrases to count

        Returns:
            pd.DataFrame: phrases and phrase scores in desc
        """
        feature_extractor = FeatureExtractor(doc=doc)
        features = feature_extractor.make_feature(
            min_length=self._min_length,
            max_length=self._max_length,
            most_common=self._most_common,
        )
        if features.shape[0] > 0:
            y_prob = self._phrase_scoring_model.predict_proba(features[:, 1:])
            phrases = feature_extractor.common_grams
            phrase_score = pd.DataFrame(
                {"phrase": list(phrases.keys()), "phrase_score": y_prob[:, 1]}
            )
            phrase_score = phrase_score.sort_values(
                by=["phrase_score"], ascending=False
            )
        else:
            phrase_score = pd.DataFrame({"phrase": [], "phrase_score": []})
        return phrase_score

    def mine_freq_phrase(self, doc: List[str]) -> pd.DataFrame:
        """main function of frequent phrase mining
        1. score phrases
        2. select phrases with domain dict
        3. form the final result dataframe

        Args:
            doc (List[str]): doc to process

        Returns:
            pd.DataFrame: final result of phrases and corresponding scores
        """
        phrase_score = self._score_phrase(doc=doc)
        self._phrase_selector = PhraseSelector(
            phrase_df=phrase_score, domain_phrase=self._domain_phrase
        )
        selected_phrases = self._phrase_selector.select_phrase()
        freq_phrase_mine_result = selected_phrases.sort_values(
            by=["phrase_score"], ascending=False
        )[: self._topn]
        freq_phrase_mine_result = freq_phrase_mine_result[
            ["phrase", "phrase_score"]
        ].sort_values(by=["phrase_score"], ascending=False)

        return freq_phrase_mine_result

    def get_phrase_idx(self, phrase_list: List[tuple], text: str) -> Dict:
        """find the location index of a given list'phrase_list' in
        the original text

        Args:
            phrase_list (List[tuple]): target words you need to find,
                                        like [('free','fire'),('cs','mode')]
            text (str): original text

        Returns:
            Dict: e.g {('free','fire'):[[1,3],[11,13]]}
        """
        tok = _tokenize(text)
        phrase_idx = _find_phrase_idx(phrase_list, tok)

        return phrase_idx
