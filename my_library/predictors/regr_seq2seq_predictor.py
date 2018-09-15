from typing import Tuple
import json

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('regr_seq2seq')
class RegrSeq2SeqPredictor(Predictor):
    """"Predictor wrapper for the Seq2Seq model"""

    def predict(self, src: str) -> JsonDict:
        """
        Predict for ``Seq2Seq`` model
        """
        return self.predict_json({"src" : src})['predicted_tokens']

    @overrides
    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        src, tgt = line.split('\t')
        return json.dumps({"src": src})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        #print(json_dict)
        decoded = json.loads(json_dict)
        src = decoded["src"]
        instance = self._dataset_reader.text_to_instance(source_string=src)

        return instance

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return outputs['predicted_vecs']