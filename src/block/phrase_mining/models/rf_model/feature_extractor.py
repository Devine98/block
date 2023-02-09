import math
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams


def _tokenize(text: str) -> List[str]:
    """tokenize text

    Args:
        text (str): text to tokenize

    Returns:
        List[str]: tokens
    """

    return re.findall(r"\w+", text)


class FeatureExtractor:
    """
    extract ngram based features, including:
        1. most common ngrams
        2. idf
        3. pmi
        4. ami = pmi/length
        5. min(left entropy, right entropy)
        6. avg(left entropy, right entropy) from smoothNLP
        7. if a certain n_gram is a subphrase of common grams, e.g
        ('free', 'fire') and ('free', 'fire', 'max')
        8. location: mean/std of indexs of phrases in original doc

    Attributes:
        doc (List[str]):
            The doc to extract features
    """

    def __init__(self, doc: List[str]) -> None:
        """
        doc: one doc is a collection of users answers for one questionnaire, one quesion
        each of an element of doc is a user's answer
        """
        self.doc = doc
        self._text = " ".join(self.doc)
        self._tokens = []
        self._n_grams_freq = {}
        self.common_grams = {}
        self._preprocess()

    def _preprocess(self) -> None:
        """
        remove stop words
        """
        stop = set(stopwords.words("english"))
        self._text = " ".join(self.doc)
        self._text = re.sub("[^-9A-Za-z ]", "", self._text).lower()
        self._tokens = [
            word
            for word in (token for token in _tokenize(self._text))
            if word not in stop
        ]
        self._text = " ".join(self._tokens)

    def _count_ngrams(self, min_length: int, max_length: int) -> None:
        """count ngrams of a given text, from min_length gram to max_length gram
        return a dict {1:1_gram_results, 2:2_gram_results,...}
        each gram results are like: {('free','fire'): 1000, ...}

        Args:
            min_length (int): min length of ngrams
            max_length (int): max length of ngrams
        """
        for i in range(min_length, max_length + 1):
            self._n_grams_freq[i] = Counter(list(ngrams(self._tokens, i)))

    def _get_most_common_ngrams(self, most_common: int) -> None:
        """get top most_common grams for each gram

        Args:
            most_common (int): the top most_common number of ngrams to count
        """
        for n in sorted(self._n_grams_freq):
            for phrase, count in self._n_grams_freq[n].most_common(most_common):
                self.common_grams[phrase] = count

    def _calculate_idf(self, phrase: Tuple[str]) -> float:
        """calculate idf value for phrase

        Args:
            phrase (Tuple[str]): phrase to calculate, e.g ['free','fire]

        Returns:
            float: idf of target phrase
        """
        num_doc = len(self.doc) + 1
        phrase = " ".join(list(phrase))
        num_phrase_in_doc = 1
        for d in self.doc:
            if phrase in d:
                num_phrase_in_doc += 1

        return math.log(num_doc / num_phrase_in_doc)

    def _calculate_pmi(self, phrase: Tuple[str]) -> Tuple[float, float]:
        """calculate pmi for one phrase

        Args:
            phrase (Tuple[str]): phrase to calculate, e.g ['free','fire]

        Returns:
            Tuple[float,float]: pmi, ami of target phrase
        """
        n = len(phrase)
        if n == 1:
            return 0, 0
        p_x_y_list = []
        for i in range(1, n):
            if (phrase[:i] in self._n_grams_freq[i]) and (
                phrase[i:] in self._n_grams_freq[n - i]
            ):
                p_x_y_list.append(
                    self._n_grams_freq[i].get(phrase[:i])
                    * self._n_grams_freq[n - i].get(phrase[i:])
                )

        p_x_y = 0.00001 if not p_x_y_list else min(p_x_y_list)
        pmi = math.log(p_x_y / (self._n_grams_freq[n][phrase] + 1))
        ami = pmi / n

        return pmi, ami

    def _find_branch_word(
        self, target_phrase: Tuple[str]
    ) -> Tuple[List[str], List[str]]:
        """for a given target phrase like ('free','fire')
        find the left words and right words of the given phrase

        e.g, for tokenized text = ['I', 'think', 'free', 'fire', 'is',
        'a', 'good', 'game'] and
        target_phrase = ('free', 'fire')
        the ouput will be ['think'], ['is']

        Args:
            target_phrase (Tuple[str]): e.g ('free', 'fire')

        Returns:
            _type_: list of left words, list of right words
        """
        phrase_idxs = _find_phrase_idx([target_phrase], self._tokens)[target_phrase]
        left_words = []
        right_words = []
        for idx in phrase_idxs:
            if idx[0] > 1:
                left_words.append(self._tokens[idx[0] - 1])
            if idx[1] < len(self._tokens) - 1:
                right_words.append(self._tokens[idx[1] + 1])

        return left_words, right_words

    def _calculate_left_right_entropy(
        self, phrases: List[Tuple[str]]
    ) -> Tuple[float, float]:
        """calculate the min/avg left right entropy of given word lists in text

        Args:
            phrases (List[Tuple[str]]): phrase to calculate

        Returns:
            Tuple[float,float]:
                min_entropy = min(left entropy, right entropy)
                avg_entropy = log((LE * e^RE + RE * e^LE)/abs(LE-RE)) from smoothNLP
        """
        min_words_entropy = {}
        avg_words_entropy = {}
        for phrase in phrases:
            left_words, right_words = self._find_branch_word(phrase)
            left_entropy = _calculate_words_entropy(left_words)

            right_entropy = _calculate_words_entropy(right_words)
            min_words_entropy[phrase] = min(left_entropy, right_entropy)
            avg_words_entropy[phrase] = self._calculate_avg_left_right_entropy(
                left_entropy, right_entropy
            )

        return min_words_entropy, avg_words_entropy

    def _calculate_avg_left_right_entropy(self, le: float, re: float) -> float:
        """
        average left and right entropy from smoothnlp
        """
        return math.log(
            (le * 2**re + re * 2**le + 0.00001) / (abs(le - re) + 1), 1.5
        )

    def _is_included(self, phrase: Tuple[str]) -> int:
        """check if a given phrase is a subphrase of a phrase in common grams
        e.g, ('free', 'fire') is a subphrase of ('free', 'fire', 'max')

        Args:
            phrase (Tuple[str]): target phrase, e.g ('free', 'fire')

        Returns:
            int: 0 or 1
        """
        is_include = 0
        for gram in self.common_grams:
            gram_str = " ".join(gram)
            phrase_str = " ".join(phrase)
            if (gram_str.find(phrase_str) == 0) and (gram_str != phrase_str):
                is_include = 1
        return is_include

    def _loc_features(self, phrase: Tuple[str]) -> Tuple[float, float]:
        """get the location of phrases,
        calculate the mean/deviation of phrases

        Args:
            phrase (Tuple[str]): target phrase

        Returns:
            Tuple[float,float]:
                mean_loc: mean of phrase locations
                dev_loc: deviation of phrase locations
        """
        loc = []
        for doc in self.doc:
            tok = _tokenize(doc)
            tmp_loc = _find_phrase_idx([phrase], tok)
            if tmp_loc:
                loc += [x[0] for x in tmp_loc.get(phrase)]

        # loc = _find_phrase_idx([phrase], self.tokens)
        # loc = [x[0] for x in loc[phrase]]
        mean_loc = np.mean(loc)
        dev_loc = np.std(loc)

        return mean_loc, dev_loc

    def make_feature(
        self, min_length: int = 1, max_length: int = 5, most_common: int = 50
    ) -> np.array:
        """calculate features, including:
        ngrams, idf, pmi, entropy, is_included for a given doc's most common grams
        and merge features together.

        Returns:
            np.array: feature matrix
        """
        self._count_ngrams(min_length=min_length, max_length=max_length)
        self._get_most_common_ngrams(most_common=most_common)

        pmi_values = {}
        ami_values = {}
        idf_values = {}
        is_include_values = {}
        mean_loc_values = {}
        dev_loc_values = {}
        self._count_ngrams(min_length=min_length, max_length=max_length)
        self._get_most_common_ngrams(most_common=most_common)
        for p in self.common_grams:
            pmi_values[p], ami_values[p] = self._calculate_pmi(phrase=p)
            mean_loc_values[p], dev_loc_values[p] = self._loc_features(phrase=p)
            idf_values[p] = self._calculate_idf(phrase=p)
            is_include_values[p] = self._is_included(phrase=p)
        (
            min_entropy_values,
            avg_entropy_values,
        ) = self._calculate_left_right_entropy(phrases=list(self.common_grams.keys()))

        # merge features
        feature = np.concatenate(
            [
                np.fromiter(self.common_grams.values(), dtype=float).reshape(-1, 1),
                np.array(
                    [math.log(x) for x in list(self.common_grams.values())],
                    dtype=float,
                ).reshape(-1, 1),
                np.fromiter(is_include_values.values(), dtype=float).reshape(-1, 1),
                np.fromiter(idf_values.values(), dtype=float).reshape(-1, 1),
                np.fromiter(pmi_values.values(), dtype=float).reshape(-1, 1),
                np.fromiter(ami_values.values(), dtype=float).reshape(-1, 1),
                np.fromiter(min_entropy_values.values(), dtype=float).reshape(-1, 1),
                np.fromiter(avg_entropy_values.values(), dtype=float).reshape(-1, 1),
                np.fromiter(mean_loc_values.values(), dtype=float).reshape(-1, 1),
                np.fromiter(dev_loc_values.values(), dtype=float).reshape(-1, 1),
            ],
            axis=1,
        )
        feature[np.isnan(feature)] = 0

        return feature


def _calculate_words_entropy(word_list: List[str]) -> float:
    """calculate the entropy of a given list of words

    Args:
        word_list (List[str]): list of target words

    Returns:
        float: entropy of the given word list
    """
    word_freq_dic = dict(Counter(word_list))
    entropy = (-1) * sum(
        [
            word_freq_dic.get(i)
            / len(word_list)
            * np.log2(word_freq_dic.get(i) / len(word_list))
            for i in word_freq_dic
        ]
    )

    return entropy


def _find_phrase_idx(phrase_list: List[tuple], token_text: List[str]) -> Dict:
    """find the location index of a given list'phrase_list' in
    the original tokenized text'token_text'

    Args:
        phrase_list (List[tuple]): target words you need to find,
                                    like [('free','fire'),('cs','mode')]
        token_text (List[str]):

    Returns:
        Dict: e.g {('free','fire'):[[1,3],[11,13]]}
    """
    phrases_idx_dict = {}
    for phrase in phrase_list:
        phrase_loc = []
        for i, word in enumerate(token_text):
            if list(phrase) == token_text[i : i + len(phrase)]:
                phrase_loc.append([i, i + len(phrase) - 1])
        if len(phrase_loc) > 0:
            phrases_idx_dict[phrase] = phrase_loc
    return phrases_idx_dict
