from typing import Dict, Union, Tuple
from prettytable import PrettyTable


class MetricTableGenerator(object):
    """
    This class processes and formats metrics provided as dictionaries. It formats these metrics into a tabular format
    for easy logging and visualization. Additionally, it tabulates per-class metrics for a specified epoch. This class
    is used for presenting model performance metrics in a structured and readable manner.
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
        overall_metric_keys = [key for key in metrics_dict.keys() if '_' not in key]

        overall_metric_table = PrettyTable()
        overall_metric_table.field_names = overall_metric_keys
        overall_metric_table.add_row([metrics_dict[key] for key in overall_metric_keys])

        per_class_metrics: Dict[str, Dict[str, str]] = {}

        for key, value in metrics_dict.items():
            if '_' in key:
                metric, class_id = key.split('_', 1)
                if class_id not in per_class_metrics:
                    per_class_metrics[class_id] = {}
                per_class_metrics[class_id][metric] = str(value)

        all_metric_keys = set()
        for metrics in per_class_metrics.values():
            all_metric_keys.update(metrics.keys())

        per_class_metric_keys = ['Class'] + list(all_metric_keys)

        per_class_metric_table = PrettyTable()
        per_class_metric_table.field_names = per_class_metric_keys

        for class_id, metrics in per_class_metrics.items():
            row = [class_id] + [metrics.get(key, 'N/A') for key in all_metric_keys]
            per_class_metric_table.add_row(row)

        return overall_metric_table, per_class_metric_table
