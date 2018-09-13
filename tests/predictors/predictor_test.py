# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import my_library

class TestPaperClassifierPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = "The string!"

        archive = load_archive('tests/fixtures/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'seq2seq')

        result = predictor.predict(inputs)

        assert len(result['predicted_tokens']) > 0
