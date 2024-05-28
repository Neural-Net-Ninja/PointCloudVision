import logging
import re
from pathlib import Path
from typing import Union

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
    This class reads a log file, finds the last improved epoch, extracts the metrics for that epoch,
    and formats and logs those metrics. It contains functions to tabulate the metrics from a log file 
    and print them in a formatted way.
    Methods:
    :param log_file_path: String to be formatted.
    :type log_file_path: str or Path
    :param per_class_metric_file_path: Path to the log for per class metrics.
    :type per_class_metric_file_path: str or Path
    :param best_epoch: Epoch with the best metrics.
    :type best_epoch: int
    """
    def __init__(self,
                 log_file_path: Union[str, Path],
                 per_class_metric_file_path: Union[str, Path],
                 best_epoch: int) -> None:

        self.log_file_path = log_file_path
        self.per_class_metric_file_path = per_class_metric_file_path
        self.best_epoch = best_epoch

    def tabulate_log_string(self) -> None:
        """Takes a log string and formats it to be printed in a tabular format.
        """
        # Read the log file
        with open(str(self.log_file_path), 'r') as file:
            log_data = file.read()

        # Find all improved epochs
        improved_epochs = re.findall(r'Epoch (\d+) improved over the previous best', log_data)

        # Get the last improved epoch
        last_improved_epoch = improved_epochs[-1]

        # Find the corresponding metrics
        pattern = r'(Epoch: \[' + re.escape(last_improved_epoch) + r'/\d+\].*?)(\n|$)'
        log_str = re.search(pattern, log_data, re.DOTALL).group(1).strip()

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

        # Log the tables
        train_title = "Training metrics:"
        test_title = "Testing metrics:"

        train_width = max(len(train_title), len(train_table.get_string().split('\n', 1)[0]))
        test_width = max(len(test_title), len(test_table.get_string().split('\n', 1)[0]))

        # Log the messages
        logger.info("\n\n%s\n%s", train_title.center(train_width), train_table)
        logger.info("\n\n%s\n%s", test_title.center(test_width), test_table)

    def tabulate_per_class_metrics(self) -> None:
        """
        Takes a log string and formats it to be printed in a tabular format.
        """
        # Read CSV file into a Pandas DataFrame
        data = pd.read_csv(str(self.per_class_metric_file_path))
        best_epoch = self.best_epoch - 1

        bar_graph = {}

        # Iterate through the header
        for column_name in data.columns:
            value = data.loc[best_epoch, column_name]
            if isinstance(value, (int, float)):
                rounded_value = round(value, 2)
                if column_name.startswith('Precision_'):
                    bar_graph[str(column_name[len('Precision_'):])] = [rounded_value]
                elif column_name.startswith('Recall_'):
                    bar_graph[str(column_name[len('Recall_'):])].append(rounded_value)
                elif column_name.startswith('Dice_'):
                    bar_graph[str(column_name[len('Dice_'):])].append(rounded_value)
                elif column_name.startswith('IoU_'):
                    bar_graph[str(column_name[len('IoU_'):])].append(rounded_value)

        table = PrettyTable()
        table.field_names = ['Class', 'Precision', 'Recall', 'Dice', 'IoU']

        for key, values in bar_graph.items():
            table.add_row([key] + values)

        title = "Per-class metrics:"
        width = max(len(title), len(table.get_string().split('\n', 1)[0]))
        logger.info("\n\n%s\n%s", title.center(width), table)


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