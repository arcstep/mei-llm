import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SOURCES = Path(__file__).resolve().parents[1] / 'sources'
sys.path.insert(0, str(SOURCES))
import tokenizer_compare as pilot


class TokenizerPilotTests(unittest.TestCase):
    def test_group_split_does_not_depend_on_text_or_chapter(self):
        a = {'source_id': 'novel', 'group_id': 'work-a', 'text': 'chapter one'}
        b = {**a, 'text': 'chapter two'}
        self.assertNotEqual(pilot.identity(a)[0], pilot.identity(b)[0])
        self.assertEqual(pilot.identity(a)[1], pilot.identity(b)[1])
        self.assertEqual(pilot.split_for(pilot.identity(a)[1], 9), pilot.split_for(pilot.identity(b)[1], 9))

    def test_identity_normalizer_preserves_parameters_and_newlines(self):
        import sentencepiece as spm
        with tempfile.TemporaryDirectory() as tmp:
            text = '{"x":"ＡＢＣ","note":"x  y","id":"00123"}\na\tb'
            spm.SentencePieceTrainer.train(sentence_iterator=iter([text] * 10),
                model_prefix=str(Path(tmp) / 'sp'), vocab_size=512, hard_vocab_limit=False,
                byte_fallback=True, normalization_rule_name='identity',
                remove_extra_whitespaces=False, add_dummy_prefix=False,
                num_threads=1, minloglevel=2)
            sp = spm.SentencePieceProcessor(model_file=str(Path(tmp) / 'sp.model'))
            self.assertEqual(sp.decode(sp.encode(text)), text)


if __name__ == '__main__':
    unittest.main()
