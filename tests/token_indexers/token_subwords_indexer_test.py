# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary

from my_library.token_indexers import TokenSubwordsIndexer
from my_library.tokenizers import BpeTokenizer

class SubwordTokenIndexerTest(AllenNlpTestCase):
    def test_count_vocab_items_respects_casing(self):
        indexer = TokenSubwordsIndexer("subwords")
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("sentencepiece"), counter)
        indexer.count_vocab_items(Token("Sentencepiece"), counter)
        assert counter["subwords"] == {"senten@@": 1, "ce@@": 2, "piece": 2, "Sen@@": 1, "ten@@": 1}

        indexer = TokenSubwordsIndexer("subwords", BpeTokenizer(lowercase_subwords=True))
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("sentencepiece"), counter)
        indexer.count_vocab_items(Token("Sentencepiece"), counter)
        assert counter["subwords"] == {"senten@@": 2, "ce@@": 2, "piece": 2}


    def test_as_array_produces_token_sequence(self):
        indexer = TokenSubwordsIndexer("subwords")
        padded_tokens = indexer.pad_token_sequence({'k': [[1, 2, 3, 4, 5], [1, 2, 3], [1]]},
                                                   desired_num_tokens={'k': 4},
                                                   padding_lengths={"num_token_subwords": 10})
        assert padded_tokens == {'k': [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                                       [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}

    def test_tokens_to_indices_produces_correct_subwords(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("piece", namespace='subwords')
        vocab.add_token_to_namespace("ce@@", namespace='subwords')
        vocab.add_token_to_namespace("c", namespace='subwords')
        vocab.add_token_to_namespace("senten@@", namespace='subwords')


        indexer = TokenSubwordsIndexer("subwords")
        indices = indexer.tokens_to_indices([Token("sentencepiece")], vocab, "bpe")
        assert indices == {"bpe": [[5, 3, 2]]}
