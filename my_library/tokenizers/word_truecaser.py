import re
from typing import List

from overrides import overrides

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token

class WordTruecaser(Registrable):
    """
    A ``WordSplitter`` splits strings into words.  This is typically called a "tokenizer" in NLP,
    because splitting strings into characters is trivial, but we use ``Tokenizer`` to refer to the
    higher-level object that splits strings into tokens (which could just be character tokens).
    So, we're using "word splitter" here for this.
    """
    default_implementation = 'moses'

    def truecase_words(self, words: List[Token]) -> List[Token]:
        """
        Returns a filtered list of words.
        """
        raise NotImplementedError


MOSES_TRUECASER_PATH = 'third_party/moses-scripts/scripts/recaser/truecase.perl'
MOSES_TRUECASING_MODEL_PATH_EN = 'models/model.tc.en'


@WordTruecaser.register('tartu')
class TartuWordTruecaser(WordTruecaser):
    """
    Truecaser that also handels words in the middle of the sentence.

    Will recase lowercased to uppercased characters if needed.
    """

    def __init__(self, moses_truecasing_model_path: str = None)  -> None:
        self._moses_truecasing_model_path = moses_truecasing_model_path or MOSES_TRUECASING_MODEL_PATH_EN
        self._tc_model = self._load_model(self._moses_truecasing_model_path)


    @overrides
    def truecase_words(self, words: List[Token]) -> List[Token]:
        sentence = [t.text for t in words]

        turecased_words = [self._tc_model[w.lower()].word if (w.lower() in self._tc_model
                and (i == 0 or self._is_upper(w) or sentence[i-1] in ".:;")) else w
                for i, w in enumerate(sentence)]
        turecased_tokens = [Token(w) for w in turecased_words]

        return turecased_tokens

    def _is_upper(self, w):
        return re.search(r'[A-Z]', w) and not re.search(r'[a-z]', w)

    def _load_model(self, filename, freqs = False):
        # truecasing func
        class DefUniqDict(dict):
            def __missing__(self, key):
                return key

        class WordFreqTuple():
            def __init__(self, word, freq):
                self.word = word
                self.freq = freq

        res = DefUniqDict()

        with open(filename, 'r') as filehandle:
            for w in filehandle:
                w, f = w.strip().split()

                res[w.lower()] = WordFreqTuple(w, int(f))

            return res
