from typing import Dict, Union, Tuple
from prettytable import PrettyTable


class MetricTabulator(object):
    """
    This class processes and formats metrics from log files. It formats these metrics into a tabular format for easy
    logging and visualization. Additionally, it tabulates per-class metrics for a specified epoch. This class is useful
    for analyzing and presenting model performance metrics in a structured and readable manner.
    """
    def tabulate_metrics(self,
                         metrics_dict: Dict[str, Union[float, int]]) -> Tuple[PrettyTable, PrettyTable]:
        """
        Takes a dictionary of metrics and formats it to be printed in a tabular format.

        :param metrics_dict: The dictionary of metrics to be formatted.
        :type metrics_dict: Dict[str, Union[float, int]]
        :return: A tuple containing two PrettyTable objects. The first table contains overall metrics, and the second
            table contains overall metrics and per-class metrics.
        :type: Tuple[PrettyTable, PrettyTable]
        """
        # Extract overall metric keys (those without a class suffix)
        overall_metric_keys = [key for key in metrics_dict.keys() if '_' not in key]

        # Create a PrettyTable object for overall metrics
        overall_metric_table = PrettyTable()
        overall_metric_table.field_names = overall_metric_keys

        # Add the values as a single row to the overall metric table
        overall_metric_table.add_row([metrics_dict[key] for key in overall_metric_keys])

        # Initialize an empty dictionary to store the extracted per-class data
        extracted_data: Dict[str, Dict[str, str]] = {}

        # Iterate over the items in the metrics_dict to extract per-class metrics
        for key, value in metrics_dict.items():
            if '_' in key:
                metric, class_id = key.split('_', 1)
                if class_id not in extracted_data:
                    extracted_data[class_id] = {}
                extracted_data[class_id][metric] = str(value)

        # Determine the per-class metric keys
        per_class_metric_keys = ['Class'] + list(next(iter(extracted_data.values())).keys())

        # Create a PrettyTable object for per-class metrics
        per_class_metric_table = PrettyTable()
        per_class_metric_table.field_names = per_class_metric_keys

        # Populate the per-class metric table with the data from the extracted_data dictionary
        for class_id, metrics in extracted_data.items():
            per_class_metric_table.add_row([
                class_id,
                metrics.get('Precision', 'N/A'),
                metrics.get('Recall', 'N/A'),
                metrics.get('Dice', 'N/A'),
                metrics.get('IoU', 'N/A')
            ])

        return overall_metric_table, per_class_metric_table