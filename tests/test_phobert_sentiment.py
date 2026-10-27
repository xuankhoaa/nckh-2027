import unittest

from phobert_sentiment import normalize_label, score_from_probabilities


class PhoBertSentimentTests(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(normalize_label("NEG"), "negative")
        self.assertEqual(normalize_label("neutral"), "neutral")
        self.assertEqual(normalize_label("LABEL_2"), "positive")

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_label("mixed")

    def test_continuous_score(self):
        self.assertAlmostEqual(score_from_probabilities(0.2, 0.7), 0.5)
        self.assertEqual(score_from_probabilities(2.0, 0.0), -1.0)


if __name__ == "__main__":
    unittest.main()
