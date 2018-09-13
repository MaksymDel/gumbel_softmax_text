from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_filter import WordFilter, PassThroughWordFilter
from allennlp.data.tokenizers.word_splitter import WordSplitter, SimpleWordSplitter
from allennlp.data.tokenizers.word_stemmer import WordStemmer, PassThroughWordStemmer


from my_library.tokenizers.moses_word_splitter import MosesWordSplitter
from my_library.tokenizers.word_truecaser import WordTruecaser, TartuWordTruecaser
from my_library.tokenizers.moses_word_filter import MosesWordFilter

@Tokenizer.register("nmt")
class NmtTokenizer(Tokenizer):
    """
    A ``NmtTokenizer`` handles an NMT specific preprocessing:
    1) Tokenizations
    2) Truecasing
    3) Filtering

    Byte pair encoding is done with separate ``BpeTokenizer`` that servers for TokenIndexing
    The class requires ``moses-scripts`` and ``subword-nmt`` repos to be cloned in the third party directory of this framework.

    Parameters
    ----------
    word_splitter : ``WordSplitter``, optional
        The :class:`WordSplitter` to use for splitting text strings into word tokens.  The default
        is to use the ``MosesWordSplitter`` with default parameters.
    word_truecaser : ``WordTruecaser``, optional
        The :class:`WordTruecaser` to use.  Default is the ``MosesWordTruecaser``.
    word_filter : ``WordFilter``, optional
        The :class:`WordFilter` to use for, e.g., removing stopwords.  Default is to use MosesWordFilter.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """
    def __init__(self,
                 word_splitter: WordSplitter = None,
                 word_truecaser: WordTruecaser = None,
                 word_filter: WordFilter = None,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        self._word_splitter = word_splitter or MosesWordSplitter()
        self._word_truecaser = word_truecaser or TartuWordTruecaser()
        self._word_filter = word_filter or MosesWordFilter()
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        Does whatever processing is required to convert a string of text into a sequence of tokens.
        At a minimum, this uses a ``WordSplitter`` to split words into text.  It may also do
        stemming or stopword removal, depending on the parameters given to the constructor.
        """
        tok_words = self._word_splitter.split_words(text)
        return self._finish_preprocessing(tok_words)

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        batched_words = self._word_splitter.batch_split_words(texts)
        return [self._finish_preprocessing(words) for words in batched_words]

    def _finish_preprocessing(self, tok_words: List[str]) -> List[Token]:
        tc_tok_words = self._word_truecaser.truecase_words(tok_words)
        filtered_tc_tok_words = self._word_filter.filter_words(tc_tok_words)

        for start_token in self._start_tokens:
            filtered_tc_tok_words.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            filtered_tc_tok_words.append(Token(end_token, -1))

        return filtered_tc_tok_words
