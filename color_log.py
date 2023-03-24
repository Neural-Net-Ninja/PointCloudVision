import logging
from termcolor import colored

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.ERROR:
            # Use red color for error messages
            record.msg = colored(record.msg, 'red')
        elif record.levelno == logging.WARNING:
            # Use orange color for warning messages
            record.msg = colored(record.msg, 'yellow')
        return super().format(record)

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.WARNING)

# Create a console handler with the custom formatter
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)

# Log an error message and a warning message
logger.error('An error occurred!')
logger.warning('Something might be wrong!')