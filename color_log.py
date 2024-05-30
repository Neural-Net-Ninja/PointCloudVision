# logger_config.py
import logging
import colorlog

def get_logger():
    # create logger
    logger = logging.getLogger(__name__)

    # set log level
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter with color scheme
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

    # add formatter to console handler
    ch.setFormatter(formatter)

    # add console handler to logger
    logger.addHandler(ch)

    return logger

# some_other_file.py
from logger_config import get_logger

logging = get_logger()

logging.debug('This is a debug message in green color!')