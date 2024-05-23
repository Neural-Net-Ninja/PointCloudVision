import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

# class TestProcessLogFile(unittest.TestCase):
#     @patch("builtins.open", new_callable=mock_open, read_data="Epoch 1 improved over the previous best\nEpoch: [1/10] Train Loss: 0.5 Train Accuracy: 0.8 || Test Loss: 0.4 Test Accuracy: 0.85")
#     def test_process_log_file(self, mock_file):
#         # Mock the tabulate_log_string function to avoid actual logging
#         with patch("tabulize.tabulate_log_string") as mock_tabulate:
#             process_log_file(Path("dummy/path"))
#             mock_file.assert_called_once_with("dummy/path", "r")
#             mock_tabulate.assert_called_once()
            
# class TestTabulateFunctions(unittest.TestCase):
#     @patch("tabulize.logger")
#     def test_tabulate_log_string(self, mock_logger):
#         log_str = "Epoch: [1/10] Train Loss: 0.5 Train Accuracy: 0.8 Train mPrecision: 0.7 Train mRecall: 0.6 Train mDice: 0.5 Train mIoU: 0.4 || Test Loss: 0.3 Test Accuracy: 0.2 Test mPrecision: 0.1 Test mRecall: 0.6 Test mDice: 0.5 Test mIoU: 0.4"
#         tabulate_log_string(log_str)
#         self.assertEqual(mock_logger.info.call_count, 2)

#     @patch("tabulize.logger")
#     @patch("tabulize.pd.read_csv")
#     def test_tabulate_per_class_metrics(self, mock_read_csv, mock_logger):
#         mock_read_csv.return_value = pd.DataFrame({
#             "Precision_class1": [0.1, 0.2],
#             "Recall_class1": [0.3, 0.4],
#             "Dice_class1": [0.5, 0.6],
#             "IoU_class1": [0.7, 0.8]
#         })
#         tabulate_per_class_metrics("dummy/path", 2)
#         mock_read_csv.assert_called_once_with("dummy/path")
#         mock_logger.info.assert_called_once()

# if __name__ == '__main__':
#     unittest.main()
    
    
# import unittest
# from unittest.mock import patch

# class TestFunctions(unittest.TestCase):
#     @patch("builtins.open", read_data="Epoch 1 improved")
#     @patch("tabulize.tabulate_log_string")
#     def test_process_log_file(self, mock_tabulate, mock_file):
#         process_log_file("dummy/path")
#         mock_file.assert_called_once()
#         mock_tabulate.assert_called_once()

#     @patch("tabulize.logger")
#     def test_tabulate_log_string(self, mock_logger):
#         tabulate_log_string("Epoch: [1/10] Train Loss: 0.5 Train Accuracy: 0.8")
#         self.assertEqual(mock_logger.info.call_count, 2)

#     @patch("tabulize.logger")
#     @patch("tabulize.pd.read_csv")
#     def test_tabulate_per_class_metrics(self, mock_read_csv, mock_logger):
#         mock_read_csv.return_value = {"Precision_class1": [0.1, 0.2]}
#         tabulate_per_class_metrics("dummy/path", 2)
#         mock_read_csv.assert_called_once()
#         mock_logger.info.assert_called_once()

# if __name__ == '__main__':
#   unittest.main()
    
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