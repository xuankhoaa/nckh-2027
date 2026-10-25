import unittest

import pandas as pd

from collect_news import (
    _month_starts,
    _parse_rss,
    score_sentiment,
    title_mentions_bank,
    validate_news_coverage,
)


class NewsCollectionTests(unittest.TestCase):
    def test_sentiment_dictionary_is_bounded_and_directional(self):
        self.assertGreater(score_sentiment("Ngân hàng báo lãi kỷ lục, vượt kế hoạch"), 0)
        self.assertLess(score_sentiment("Lợi nhuận giảm, nợ xấu tăng và bị xử phạt"), 0)
        self.assertEqual(score_sentiment("Ngân hàng công bố lịch họp đại hội cổ đông"), 0)
        self.assertTrue(-1 <= score_sentiment("lãi tăng nhưng nợ xấu tăng") <= 1)

    def test_month_intervals_cover_period_without_gap(self):
        intervals = _month_starts(pd.Timestamp("2023-01-15"), pd.Timestamp("2023-03-02"))
        self.assertEqual(intervals[0][0], pd.Timestamp("2023-01-15"))
        self.assertEqual(intervals[-1][1], pd.Timestamp("2023-03-03"))
        for first, second in zip(intervals, intervals[1:]):
            self.assertEqual(first[1], second[0])

    def test_rss_parser_removes_source_suffix(self):
        xml = b"""<?xml version='1.0' encoding='UTF-8'?>
        <rss><channel><item><title>VCB loi nhuan tang - Bao A</title>
        <link>https://news.google.com/a</link><pubDate>Wed, 01 Jan 2025 08:00:00 GMT</pubDate>
        <source url='https://example.com'>Bao A</source></item></channel></rss>"""
        records = _parse_rss(xml, "VCB", "Vietcombank")
        self.assertEqual(records[0]["title"], "VCB loi nhuan tang")
        self.assertEqual(records[0]["source"], "Bao A")

    def test_title_must_name_the_correct_bank(self):
        self.assertTrue(title_mentions_bank("Vietcombank công bố kết quả kinh doanh", "VCB"))
        self.assertTrue(title_mentions_bank("Ngân hàng ACB chia cổ tức", "ACB"))
        self.assertFalse(title_mentions_bank("Lãi suất ngân hàng hôm nay", "VCB"))

    def test_coverage_requires_all_three_splits(self):
        frame = pd.DataFrame(
            {
                "ticker": ["VCB", "VCB", "VCB"],
                "published_at": [
                    "2023-06-01T08:00:00+07:00",
                    "2024-06-01T08:00:00+07:00",
                    "2025-06-01T08:00:00+07:00",
                ],
            }
        )
        coverage = validate_news_coverage(frame, ["VCB"], minimum_per_split=1)
        self.assertEqual(coverage.loc["VCB"].tolist(), [1, 1, 1])
        with self.assertRaises(ValueError):
            validate_news_coverage(frame, ["VCB"], minimum_per_split=2)


if __name__ == "__main__":
    unittest.main()
