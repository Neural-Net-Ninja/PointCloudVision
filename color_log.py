import logging
import colorlog

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

# log a debug message
logger.debug('This is a debug message in green color!')
logger.error('An error occurred!')
logger.warning('Something might be wrong!')
logger.critical('A critical error occurred!')
logger.info('This is an info message in white color!')
