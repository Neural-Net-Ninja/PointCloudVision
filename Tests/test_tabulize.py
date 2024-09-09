import unittest
from unittest.mock import patch
from prettytable import PrettyTable
from typing import Dict, Union


class TestMetricTabulator(unittest.TestCase):
    """
    TestCase for the MetricTabulator class. This mostly covers correctly tabulating and formatting metrics into tables
    when the tabulate_metrics method is called.
    """
    @classmethod
    def tearDown(self):
        """
        Reset the logging configuration after each test.
        """
        reset_logging()

    def setUp(self):
        self.tabulator = MetricTabulator()
        self.metrics_dict = {
            "Epoch": 1,
            "Loss": 0.25,
            "Accuracy": 0.95,
            "mPrecision": 0.9,
            "mRecall": 0.85,
            "mDice": 0.8,
            "mIoU": 0.75,
            "Precision_Class1": 0.9,
            "Recall_Class1": 0.85,
            "Dice_Class1": 0.8,
            "IoU_Class1": 0.75,
            "Precision_Class2": 0.88,
            "Recall_Class2": 0.83,
            "Dice_Class2": 0.78,
            "IoU_Class2": 0.73
        }

    def test_tabulate_metrics(self):
        metric_table, per_class_metric_table = self.tabulator.tabulate_metrics(self.metrics_dict)

        # Verify the main metric table
        expected_main_table = PrettyTable()
        expected_main_table.field_names = ["Epoch", "Loss", "Accuracy", "mPrecision", "mRecall", "mDice", "mIoU"]
        expected_main_table.add_row([1, 0.25, 0.95, 0.9, 0.85, 0.8, 0.75])
        self.assertEqual(metric_table.get_string(), expected_main_table.get_string())

        # Verify the per-class metric table
        expected_per_class_table = PrettyTable()
        expected_per_class_table.field_names = ['Class', 'Precision', 'Recall', 'Dice', 'IoU']
        expected_per_class_table.add_row(['Class1', '0.9', '0.85', '0.8', '0.75'])
        expected_per_class_table.add_row(['Class2', '0.88', '0.83', '0.78', '0.73'])
        self.assertEqual(per_class_metric_table.get_string(), expected_per_class_table.get_string())

