from typing import Dict, List
import itertools

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from my_library.tokenizers import BpeTokenizer


@TokenIndexer.register("subwords")
class TokenSubwordsIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents tokens as lists of subword indices.
    Parameters
    ----------
    namespace : ``str``, optional (default=``token_subwords``)
        We will use this namespace in the :class:`Vocabulary` to map the subwords in each token
        to indices.
    subword_tokenizer : ``BpeTokenizer``, optional (default=``BpeTokenizer()``)
        We use a :class:`BpeTokenizer` to handle splitting tokens into subwords, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``BpeTokenizer`` with its default parameters, which uses unicode subwords and
        retains casing.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'token_subwords',
                 bpe_tokenizer: BpeTokenizer = BpeTokenizer()) -> None:
        self._namespace = namespace
        self._bpe_tokenizer = bpe_tokenizer

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('TokensubwordsIndexer needs a tokenizer that retains text')
        for subword in self._bpe_tokenizer.tokenize(token.text):
            # If `text_id` is set on the subword token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this subword.
            if getattr(subword, 'text_id', None) is None:
                counter[self._namespace][subword.text] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        indices: List[List[int]] = []
        for token in tokens:
            token_indices: List[int] = []
            if token.text is None:
                raise ConfigurationError('TokensubwordsIndexer needs a tokenizer that retains text')
            for subword in self._bpe_tokenizer.tokenize(token.text):
                if getattr(subword, 'text_id', None) is not None:
                    # `text_id` being set on the token means that we aren't using the vocab, we just
                    # use this id instead.
                    index = subword.text_id
                else:
                    index = vocabulary.get_token_index(subword.text, self._namespace)
                token_indices.append(index)
            indices.append(token_indices)
        return {index_name: indices}

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        return {'num_token_subwords': len(token)}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[List[int]]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[List[int]]]:
        # Pad the tokens.
        # tokens has only one key...
        key = list(tokens.keys())[0]

        padded_tokens = pad_sequence_to_length(
                tokens[key], desired_num_tokens[key],
                default_value=self.get_padding_token
        )

        # Pad the subwords within the tokens.
        desired_token_length = padding_lengths['num_token_subwords']
        longest_token: List[int] = max(tokens[key], key=len, default=[])
        padding_value = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" so we can take advantage of the fast implementation of itertools.zip_longest.
            padded_tokens.append([padding_value] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_value)))
        if desired_token_length > len(longest_token):
            # Removes the "dummy token".
            padded_tokens.pop()
        # Truncates all the tokens to the desired length, and return the result.
        return {key: [list(token[:desired_token_length]) for token in padded_tokens]}