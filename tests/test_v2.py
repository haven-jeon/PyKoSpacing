"""KoSpacing v2 tests. Skipped unless PYKOSPACING_V2_CHECKPOINT points to a checkpoint (and torch is installed).

    PYKOSPACING_V2_CHECKPOINT=<dir> KOSPACING_ROOT=<kospacing> python -m unittest tests.test_v2
"""
import os
import sys
import unittest

from pykospacing.v2 import split_long

CHECKPOINT = os.environ.get('PYKOSPACING_V2_CHECKPOINT')


class SplitTest(unittest.TestCase):
    def test_short_text_is_one_piece(self):
        self.assertEqual(split_long('가나다', 10), ['가나다'])

    def test_cuts_after_sentence_end(self):
        pieces = split_long('가나다라. 마바사아자. 차카타파하.', 12)
        self.assertEqual(''.join(pieces), '가나다라. 마바사아자. 차카타파하.')
        self.assertTrue(all(len(p) <= 12 for p in pieces))
        self.assertTrue(pieces[0].endswith('.'))

    def test_hard_cut_without_boundaries(self):
        pieces = split_long('가' * 25, 10)
        self.assertEqual([len(p) for p in pieces], [10, 10, 5])


@unittest.skipUnless(CHECKPOINT, 'PYKOSPACING_V2_CHECKPOINT not set')
class V2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pykospacing import Spacing
        cls.spacing = Spacing(backend='v2', checkpoint=CHECKPOINT)

    def test_no_tensorflow(self):
        self.assertNotIn('tensorflow', sys.modules)

    def test_space(self):
        self.assertEqual(self.spacing('아버지가방에들어가신다'), '아버지가 방에 들어가신다')

    def test_characters_are_kept(self):
        text = "김형호영화시장분석가는'1987'의네이버영화정보를R과KoNLP로분석했다."
        self.assertEqual(self.spacing(text).replace(' ', ''), text)

    def test_batch(self):
        out = self.spacing(['아버지가방에들어가신다', '오늘날씨가좋다'])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], '아버지가 방에 들어가신다')

    def test_fix_keeps_characters(self):
        text = '오늘은날씨가 정말좋 네요.'
        self.assertEqual(self.spacing.fix(text).replace(' ', ''), text.replace(' ', ''))

    def test_space_probs(self):
        probs = self.spacing.space_probs('아버지가 방에')
        self.assertEqual(''.join(c for c, _ in probs), '아버지가방에')
        self.assertTrue(all(0.0 <= p <= 1.0 for _, p in probs))

    def test_rules(self):
        from pykospacing import Spacing
        spacing = Spacing(backend='v2', checkpoint=CHECKPOINT, rules=['구레나룻'])
        self.assertIn('구레나룻', spacing('귀밑에서턱까지잇따라난수염을구레나룻이라고한다.'))

    def test_long_input_keeps_characters(self):
        text = '이것은긴문장의시험입니다.' * 50
        self.assertEqual(self.spacing(text).replace(' ', ''), text)

    def test_tf_only_options(self):
        with self.assertRaises(ValueError):
            self.spacing('가나다', ignore='pre')


class V1Test(unittest.TestCase):
    def test_unknown_backend(self):
        from pykospacing import Spacing
        for name in ('tf', 'tagger', 'v3'):
            with self.assertRaises(ValueError):
                Spacing(backend=name)

    def test_fix_needs_v2(self):
        from pykospacing.kospacing import Spacing
        spacing = Spacing.__new__(Spacing)
        spacing._v2 = None
        with self.assertRaises(NotImplementedError):
            spacing.fix('가나다')


if __name__ == '__main__':
    unittest.main()
