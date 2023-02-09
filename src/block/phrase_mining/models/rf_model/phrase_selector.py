from typing import List

import pandas as pd

from block.phrase_mining.models.rf_model.feature_extractor import _tokenize


class PhraseSelector:
    """
    1. if phrase in domain phrase stay
    2. filter pure number, single character

    Attributes:
        phrase_df (pd.DataFrame):
            pandas dataframe for phrases and frequency.
        domain_phrase (List[str]):
            List of in domain phrases.
    """

    def __init__(self, phrase_df: pd.DataFrame, domain_phrase: List[str]) -> None:
        self._phrase_df = phrase_df
        self._domain_phrase = [tuple(_tokenize(x.lower())) for x in domain_phrase]

    def _domain_filter(self):
        """domain phrase is a list of 1400+ phrases that ops. offers;
        if extracted phrase is in the domain list, it will stay
        """
        if "phrase_score" in self._phrase_df.columns:
            self._phrase_df.loc[
                self._phrase_df.phrase.isin(self._domain_phrase), "phrase_score"
            ] += 1
        else:
            pass

    def _number_filter(self):
        """
        filter pure number phrases, e.g ('9',)
        """
        if ("phrase" and "phrase_score") in self._phrase_df.columns:
            self._phrase_df["phrase_str"] = self._phrase_df["phrase"].apply(
                lambda x: "".join(x)
            )
            self._phrase_df.loc[
                self._phrase_df.phrase_str.astype(str).str.isnumeric(), "phrase_score"
            ] -= 1
        else:
            pass

    def select_phrase(self):
        self._domain_filter()
        self._number_filter()

        return self._phrase_df


if __name__ == "__main__":
    phrase_df = pd.DataFrame({"phrase": [], "phrase_score": []})
    phrase_df = pd.DataFrame(
        {"phrase": [("free", "fire"), ("999", "888")], "phrase_score": [0.9, 0.1]}
    )
    phrase_df = pd.DataFrame()
    domain_phrase = ["free fire"]
    ps = PhraseSelector(phrase_df=phrase_df, domain_phrase=domain_phrase)
    ps._domain_filter()
    ps._number_filter()
    res = ps.select_phrase()
