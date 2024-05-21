import pandas as pd
import re
from pathlib import Path
from typing import Optional, Union
from prettytable import PrettyTable
import logging


def tabulate_log_string(log_str: str) -> None:
    """Takes a log string and formats it to be printed in a tabular format.

    :param log_str: string to be formatted
    :type log_str: string
    :return: formatted string
    :rtype: string
    """
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
    logging.info("Training matrics:\n%s", train_table)
    logging.info("Testing matrics:\n%s", test_table)


def tabulate_per_class_matrics(log_path: Optional[Union[str, Path]], best_epoch: int) -> None:
    """
    Takes a log string and formats it to be printed in a tabular format.

    :param log_path: Path to the log file.
    :type log_path: string, Path. optional.
    :param best_epoch: Epoch with the best metrics.
    :type best_epoch: integer
    """
    # Read CSV file into a Pandas DataFrame
    data = pd.read_csv(str(log_path))
    best_epoch = best_epoch - 1

    bar_graph = {}

    # Iterate through the header
    for column_name in data.columns:
        if column_name.startswith('Precision_'):
            bar_graph[str(column_name[len('Precision_'):])] = [round(data.loc[best_epoch, column_name], 2)]
        elif column_name.startswith('Recall_'):
            bar_graph[str(column_name[len('Recall_'):])].append(round(data.loc[best_epoch, column_name], 2))
        elif column_name.startswith('Dice_'):
            bar_graph[str(column_name[len('Dice_'):])].append(round(data.loc[best_epoch, column_name], 2))
        elif column_name.startswith('IoU_'):
            bar_graph[str(column_name[len('IoU_'):])].append(round(data.loc[best_epoch, column_name], 2))

    table = PrettyTable()
    table.field_names = ['Class', 'Precision', 'Recall', 'Dice', 'IoU']

    for key, values in bar_graph.items():
        table.add_row([key] + values)

    title = "Per-class metrics:"
    width = max(len(title), len(table.get_string().split('\n', 1)[0]))
    logging.info("\n\n%s\n%s", title.center(width), table)
    

import re

# Read the log file
with open('Q:/50Hertz/Machine_learning/test_2024_predictions/ML_test_2024/03_log/log_train.txt', 'r') as file:
    log_data = file.read()

# Find all improved epochs
improved_epochs = re.findall(r'Epoch (\d+) improved over the previous best', log_data)

# Get the last improved epoch
last_improved_epoch = improved_epochs[-1]

# Find the corresponding metrics
pattern = r'(Epoch: \[' + re.escape(last_improved_epoch) + r'/\d+\].*?)(\n|$)'
metrics = re.search(pattern, log_data, re.DOTALL).group(1).strip()

tabulate_log_string(metrics)

print(metrics)



#tabulate_log_string_1(metrics)