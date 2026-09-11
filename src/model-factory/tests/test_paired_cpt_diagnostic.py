import unittest

from evaluation.base.paired_cpt_diagnostic import selected_indices, weighted_loss


class PairedDiagnosticTests(unittest.TestCase):
    def test_stratified_selection_covers_late_windows_without_duplicates(self):
        indices = selected_indices(1226, 128)
        self.assertEqual(len(indices), 128)
        self.assertEqual(len(set(indices)), 128)
        self.assertEqual(indices[0], 0)
        self.assertGreater(indices[-1], 1200)
        self.assertEqual(selected_indices(3, 0), [0, 1, 2])


    def test_token_weighted_loss_handles_short_tail(self):
        rows = [{"nll_sum": 20.0, "predicted_tokens": 10},
                {"nll_sum": 8.0, "predicted_tokens": 1}]
        self.assertAlmostEqual(weighted_loss(rows), 28 / 11)
        self.assertNotAlmostEqual(weighted_loss(rows), 5.0)
        with self.assertRaises(ValueError):
            weighted_loss([])
