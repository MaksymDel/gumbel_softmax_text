from typing import List
import subprocess

from overrides import overrides

from allennlp.data.tokenizers.token import Token

from allennlp.data.tokenizers.word_splitter import WordSplitter

MOSES_TOKENIZER_PATH = 'third_party/moses-scripts/scripts/tokenizer/tokenizer.perl'

@WordSplitter.register('moses')
class MosesWordSplitter(WordSplitter):
    """
    Wrapper for .perl Moses tokeinzer.

     Requires `moses-scripts` repo to be cloned into `third_party` folder of this framework.
    """


    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        """
        Splits a sentence into word tokens.  We handle four kinds of things: words with punctuation
        that should be ignored as a special case (Mr. Mrs., etc.), contractions/genitives (isn't,
        don't, Matt's), and beginning and ending punctuation ("antennagate", (parentheticals), and
        such.).
        The basic outline is to split on whitespace, then check each of these cases.  First, we
        strip off beginning punctuation, then strip off ending punctuation, then strip off
        contractions.  When we strip something off the beginning of a word, we can add it to the
        list of tokens immediately.  When we strip it off the end, we have to save it to be added
        to after the word itself has been added.  Before stripping off any part of a token, we
        first check to be sure the token isn't in our list of special cases.
        """

        pipe = subprocess.Popen(["perl", MOSES_TOKENIZER_PATH, sentence], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        pipe.stdin.write(sentence.encode("utf-8"))
        pipe.stdin.close()
        tokenized_sentence = pipe.stdout.read()
        tokenized_sentence = tokenized_sentence.strip().decode()

        tokens = [Token(t) for t in tokenized_sentence.split()]

        return tokens
