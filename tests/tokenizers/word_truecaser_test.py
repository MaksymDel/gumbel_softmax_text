# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token

from my_library.tokenizers import TartuWordTruecaser

class TestMosesWordTruecaser():
    def test_truecaser(self):
        truecaser = TartuWordTruecaser()

        words = [Token(w) for w in "He is the sentence !".split(" ")]
        tokens = truecaser.truecase_words(words)
        print("TOKENS: ", tokens)
        assert len(tokens) == len(words)
        assert [t.text for t in tokens] == ["he", "is", "the", "sentence", "!"]


        words = [Token(w) for w in "NASA is the sentence !".split()]
        tokens = truecaser.truecase_words(words)
        print("TOKENS: ", tokens)
        assert len(tokens) == len(words)
        assert [t.text for t in tokens] == ["NASA", "is", "the", "sentence", "!"]
