import logging
from prettytable import PrettyTable

# Sample dictionary
data = {"Epoch": 1,
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
            "IoU_Class2": 0.73}

# Configure logging
logging.basicConfig(level=logging.INFO)

# Log dictionary as plain text
logging.info(f"Data: {data}")

from prettytable import PrettyTable


# Flatten the dictionary for PrettyTable
flattened_data = [(k, v if not isinstance(v, dict) else ', '.join(f'{sub_k}: {sub_v}' for sub_k, sub_v in v.items()))
                  for k, v in data.items()]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a PrettyTable object
table = PrettyTable()
table.field_names = ["Key", "Value"]

# Add rows to the table
for key, value in flattened_data:
    table.add_row([key, value])

# Log the table
logging.info(f"\n{table}")

