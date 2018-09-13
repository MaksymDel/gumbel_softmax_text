# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list

from my_library.dataset_readers import MtDatasetReader

class TestMtDatasetReader():
    @pytest.mark.parametrize("lazy", (True, False))
    def test_default_format(self, lazy):
        reader = MtDatasetReader(lazy=lazy)
        instances = reader.read('tests/fixtures/data/mt_copy.tsv')
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["src_tokens"].tokens] == ["@start@", "this", "is",
                                                                    "a", "sentence", "with", "NASA", "@end@"]
        assert [t.text for t in fields["tgt_tokens"].tokens] == ["@start@", "this", "is",
                                                                    "a", "sentence", "with", "NASA", "@end@"]
        fields = instances[1].fields
        assert [t.text for t in fields["src_tokens"].tokens] == ["@start@", "English", "sentence",
                                                                    "again", "@end@"]
        assert [t.text for t in fields["tgt_tokens"].tokens] == ["@start@", "English", "sentence",
                                                                    "again", "@end@"]
        fields = instances[2].fields
        assert [t.text for t in fields["src_tokens"].tokens] == ["@start@", "all", "these", "sentences",
                                                                    "should", "get", "copied", "!", "@end@"]
        assert [t.text for t in fields["tgt_tokens"].tokens] == ["@start@", "all", "these", "sentences",
                                                                    "should", "get", "copied", "!", "@end@"]

    def test_source_add_start_token(self):
        reader = MtDatasetReader(source_add_start_token=False)
        instances = reader.read('tests/fixtures/data/mt_copy.tsv')
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["src_tokens"].tokens] == ["this", "is", "a", "sentence", "with", "NASA", "@end@"]
        assert [t.text for t in fields["tgt_tokens"].tokens] == ["@start@", "this", "is",
                                                                    "a", "sentence", "with", "NASA", "@end@"]