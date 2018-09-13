# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

from my_library.tokenizers import MosesWordSplitter

class TestMosesWordSplitter():
    def test_splitter(self):
        splitter = MosesWordSplitter()

        sentence = "I'am the sentence!"

        tokens = splitter.split_words(sentence)

        print("TOKENS: ", tokens)
        assert len(tokens) == 5 # I &apos;am the sentence !

        assert [t.text for t in tokens] == ["I", "&apos;am", "the", "sentence", "!"]
