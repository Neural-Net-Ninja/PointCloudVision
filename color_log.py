import logging
import colorlog

def get_logger(name: str = __name__, level: int = logging.DEBUG) -> logging.Logger:
    """
    Creates and configures a logger with colored output.

    :param name: The name of the logger. Defaults to the module's name.
    :type name: str
    :param level: The logging level. Defaults to logging.DEBUG.
    :type level: int
    :return: Configured logger instance.
    :rtype: logging.Logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter with color scheme
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s:%(message)s',
        log_colors={
            'DEBUG': 'green',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger

# Usage example in some_other_file.py
if __name__ == "__main__":
    logger = get_logger()
    logger.debug('This is a debug message in green color!')