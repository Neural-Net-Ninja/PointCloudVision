import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

class TestProcessLogFile(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data="Epoch 1 improved over the previous best\nEpoch: [1/10] Train Loss: 0.5 Train Accuracy: 0.8 || Test Loss: 0.4 Test Accuracy: 0.85")
    def test_process_log_file(self, mock_file):
        # Mock the tabulate_log_string function to avoid actual logging
        with patch("tabulize.tabulate_log_string") as mock_tabulate:
            process_log_file(Path("dummy/path"))
            mock_file.assert_called_once_with("dummy/path", "r")
            mock_tabulate.assert_called_once()
            
class TestTabulateFunctions(unittest.TestCase):
    @patch("tabulize.logger")
    def test_tabulate_log_string(self, mock_logger):
        log_str = "Epoch: [1/10] Train Loss: 0.5 Train Accuracy: 0.8 Train mPrecision: 0.7 Train mRecall: 0.6 Train mDice: 0.5 Train mIoU: 0.4 || Test Loss: 0.3 Test Accuracy: 0.2 Test mPrecision: 0.1 Test mRecall: 0.6 Test mDice: 0.5 Test mIoU: 0.4"
        tabulate_log_string(log_str)
        self.assertEqual(mock_logger.info.call_count, 2)

    @patch("tabulize.logger")
    @patch("tabulize.pd.read_csv")
    def test_tabulate_per_class_metrics(self, mock_read_csv, mock_logger):
        mock_read_csv.return_value = pd.DataFrame({
            "Precision_class1": [0.1, 0.2],
            "Recall_class1": [0.3, 0.4],
            "Dice_class1": [0.5, 0.6],
            "IoU_class1": [0.7, 0.8]
        })
        tabulate_per_class_metrics("dummy/path", 2)
        mock_read_csv.assert_called_once_with("dummy/path")
        mock_logger.info.assert_called_once()

if __name__ == '__main__':
    unittest.main()