import logging
import re
from pathlib import Path
from typing import Union

import unittest
from unittest.mock import patch, mock_open, MagicMock


class TestMetricTabulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp_dir = env.get_temporary_folder_path() / cls.__name__

    def tearDown(self):
        # need to reset all of the logging handles, use function for this
        reset_logging()

    @patch("builtins.open", new_callable=mock_open, read_data="Epoch 1 improved over the previous best\nEpoch: [1/10] Train Loss: 0.5 Train Accuracy: 0.8 Train mPrecision: 0.7 Train mRecall: 0.6 Train mDice: 0.5 Train mIoU: 0.4 || Test Loss: 0.3 Test Accuracy: 0.7 Test mPrecision: 0.6 Test mRecall: 0.5 Test mDice: 0.4 Test mIoU: 0.3")
    @patch("logging.Logger.info")
    def test_tabulate_log_string(self, mock_logger_info, mock_file):
        tabulator = MetricTabulator()
        tabulator.log_file_path = self._temp_dir / 'logs/log_train.txt'
        tabulator.tabulate_log_string()

        # Check if the logger.info method was called with the expected arguments.
        mock_logger_info.assert_called()

if __name__ == '__main__':
    from context import set_path

    set_path()
    from test.context import MetricTabulator, env, reset_logging

    env.setup_environment()
    unittest.main(exit=False)
    env.delete_temporary_folder()
else:
    # due to the tests being executed using multiprocessing, which on Windows spawns a process named __mp_main__, and
    # the autodoc module importing the files it documents, the relative imports need to be strictly guarded to be only
    # done if the test is executed as such, otherwise they cause errors
    if __name__ in 'test.metrics.test_part_metric_tabulator':
        from test.context import MetricTabulator, env, reset_logging


if __name__ == '__main__':
  unittest.main()
    
from prettytable import PrettyTable

# Specify the Column Names while initializing the Table
myTable = PrettyTable(["Student Name", "Class", "Section", "Percentage"])

# Add rows
myTable.add_row(["Leonard", "X", "B", "91.2"])
myTable.add_row(["Penny", "X", "C", "63.5"])
myTable.add_row(["Howard", "X", "A", "85.6"])
myTable.add_row(["Sheldon", "X", "A", "99.1"])

print(myTable)

table = PrettyTable()

# Add columns
table.field_names = ["City name", "Area", "Population", "Annual Rainfall"]

# Align city names to the left
table.align["City name"] = "l"

table.add_row(["Adelaide", 1295, 1158259, 600.5])
table.add_row(["Brisbane", 5905, 1857594, 1146.4])
table.add_row(["Darwin", 112, 120900, 1714.7])
table.add_row(["Hobart", 1357, 205556, 619.5])
table.add_row(["Sydney", 2058, 4336374, 1214.8])
table.add_row(["Melbourne", 1566, 3806092, 646.9])
table.add_row(["Perth", 5386, 1554769, 869.4])

print(table)

table = PrettyTable()

table.field_names = ["City name", "Area", "Population", "Annual Rainfall"]

table.add_row(["Adelaide", 1295, 1158259, 600.5])
table.add_row(["Brisbane", 5905, 1857594, 1146.4])
table.add_row(["Darwin", 112, 120900, 1714.7])
table.add_row(["Hobart", 1357, 205556, 619.5])
table.add_row(["Sydney", 2058, 4336374, 1214.8])
table.add_row(["Melbourne", 1566, 3806092, 646.9])
table.add_row(["Perth", 5386, 1554769, 869.4])

# Sort by population
table.sortby = "Population"

print(table)