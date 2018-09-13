from typing import List

from overrides import overrides

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_filter import WordFilter

@WordFilter.register('moses')
class MosesWordFilter(WordFilter):
    """
    Does not filter words; it's a no-op.  This is the default word filter.

    In the future something like Moses's clean-corpus.perl should be implemented.
    Probably will throw "Strange or empty line" exception that is captured in DatasetReader.
    Needed to skip both src and tgt sentence.
    """
    @overrides
    def filter_words(self, words: List[Token]) -> List[Token]:
        return words