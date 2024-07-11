import logging
import re
from typing import List

import pandas as pd
from prettytable import PrettyTable

# Create a new logger
logger = logging.getLogger('metrics_tabulator_logger')
logger.propagate = False

# Add a handler to the logger if it doesn't have one
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


class MetricTabulator:
    """
    This class is responsible for processing and formatting metrics from log files. It provides methods to:

    1. Parse log strings to extract training and testing metrics.
    2. Format these metrics into a tabular format for easy logging and visualization.
    3. Handle both standard and "coarse" metrics if present in the log data.
    4. Tabulate per-class metrics for a specified epoch.

    This class is useful for analyzing and presenting model performance metrics in a structured and readable manner.
    """
    def tabulate_log_string(self, log_data: str) -> None:
        """
        Parses a log string to extract training and testing metrics, and formats them into a tabular format for logging.

        The method uses regular expressions to extract metrics such as loss, accuracy, precision, recall, dice, and IoU
        for both training and testing phases. If the log string contains "coarse" metrics, it will also extract those.

        :param log_data: The log string to be formatted.
        :type log_data: str
        :return: None
        """
        log_str = log_data.strip()

        # Define the regular expression pattern to extract the values for train and test
        if "coarse" in log_str:
            pattern = (
                r".*Epoch: \[(\d+)/(\d+)\] "
                r"Train Loss: (\d+\.\d+) "
                r"Train Accuracy: (\d+\.\d+) "
                r"Train mPrecision: (\d+\.\d+) "
                r"Train mRecall: (\d+\.\d+) "
                r"Train mDice: (\d+\.\d+) "
                r"Train mIoU: (\d+\.\d+) "
                r"Train Accuracy_coarse: (\d+\.\d+) "
                r"Train mPrecision_coarse: (nan|\d+\.\d+) "
                r"Train mRecall_coarse: (nan|\d+\.\d+) "
                r"Train mDice_coarse: (nan|\d+\.\d+) "
                r"Train mIoU_coarse: (nan|\d+\.\d+) "
                r"\|\| Test Loss: (\d+\.\d+) "
                r"Test Accuracy: (\d+\.\d+) "
                r"Test mPrecision: (\d+\.\d+) "
                r"Test mRecall: (\d+\.\d+) "
                r"Test mDice: (\d+\.\d+) "
                r"Test mIoU: (\d+\.\d+) "
                r"Test Accuracy_coarse: (\d+\.\d+) "
                r"Test mPrecision_coarse: (nan|\d+\.\d+) "
                r"Test mRecall_coarse: (nan|\d+\.\d+) "
                r"Test mDice_coarse: (nan|\d+\.\d+) "
                r"Test mIoU_coarse: (nan|\d+\.\d+)"
            )
        else:
            pattern = (
                r".*Epoch: \[(\d+)/(\d+)\] "
                r"Train Loss: (\d+\.\d+) "
                r"Train Accuracy: (\d+\.\d+) "
                r"Train mPrecision: (\d+\.\d+) "
                r"Train mRecall: (\d+\.\d+) "
                r"Train mDice: (\d+\.\d+) "
                r"Train mIoU: (\d+\.\d+) "
                r"\|\| Test Loss: (\d+\.\d+) "
                r"Test Accuracy: (\d+\.\d+) "
                r"Test mPrecision: (\d+\.\d+) "
                r"Test mRecall: (\d+\.\d+) "
                r"Test mDice: (\d+\.\d+) "
                r"Test mIoU: (\d+\.\d+)"
            )

        # Create the train and test tables
        train_table = PrettyTable()
        train_table.field_names = ["Epoch", "Loss", "Accuracy", "mPrecision", "mRecall",
                                    "mDice", "mIoU"]
        test_table = PrettyTable()
        test_table.field_names = ["Epoch", "Loss", "Accuracy", "mPrecision", "mRecall",
                                    "mDice", "mIoU"]

        # Extract the values from the string and add them to the tables
        match = re.match(pattern, log_str)
        if match:
            train_values = [int(match.group(1)), float(match.group(3)), float(match.group(4)), float(match.group(5)),
                            float(match.group(6)), float(match.group(7)), float(match.group(8))]
            test_values = [int(match.group(1)), float(match.group(9)), float(match.group(10)), float(match.group(11)),
                            float(match.group(12)), float(match.group(13)), float(match.group(14))]
            train_table.add_row(train_values)
            test_table.add_row(test_values)
        else:
            logging.warning("No match found for last improved epoch, aborting log tabulation.")
            return

        # Log the tables
        train_title = "Training metrics:"
        test_title = "Testing metrics:"

        # Log the messages
        logger.info("%s\n%s", train_title, train_table)
        logger.info("%s\n%s", test_title, test_table)

    def tabulate_per_class_metrics(self,
                                   best_epoch: int,
                                   per_class_metric: List[float],
                                   col_names: List[str]) -> None:
        """
        Takes a log string and formats it to be printed in a tabular format.

        :param best_epoch: The epoch number of the best model.
        :type best_epoch: int
        :param per_class_metric: The per class metrics to be formatted.
        :type per_class_metric: List[float]
        :param col_names: The column names for the per class metrics.
        :type col_names: List[str]
        :return: None
        """
        # Add best epoch to the per class metrics list
        per_class_metric = [best_epoch] + per_class_metric

        # Elements to remove
        elements_to_remove = ['Num_samples', 'Timestamp', 'Elapsed_time_(s)']

        # Remove specified elements from the list
        col_names = [col for col in col_names if col not in elements_to_remove]

        data = pd.DataFrame([per_class_metric], columns=col_names)

        metrics_by_class = {}

        # Iterate through the header
        for column_name in data.columns:
            value = data.loc[best_epoch - 1, column_name]
            if isinstance(value, (int, float)):
                rounded_value = f"{round(value, 1):.2f}"
                if column_name.startswith('Precision_'):
                    metrics_by_class[str(column_name[len('Precision_'):])] = [rounded_value]
                elif column_name.startswith('Recall_'):
                    key = str(column_name[len('Recall_'):])
                    metrics_by_class.setdefault(key, []).append(rounded_value)
                elif column_name.startswith('Dice_'):
                    key = str(column_name[len('Dice_'):])
                    metrics_by_class.setdefault(key, []).append(rounded_value)
                elif column_name.startswith('IoU_'):
                    key = str(column_name[len('IoU_'):])
                    metrics_by_class.setdefault(key, []).append(rounded_value)

        table = PrettyTable()
        table.field_names = ['Class', 'Precision', 'Recall', 'Dice', 'IoU']

        for key, values in metrics_by_class.items():
            table.add_row([key] + values)

        title = "Per-class metrics:"
        logger.info("%s\n%s", title, table)
