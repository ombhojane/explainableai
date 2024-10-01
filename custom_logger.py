import logging
from colorama import Fore, Style

class CustomFormatter(logging.Formatter):
    """Custom formatter to add color to log messages based on their severity."""

    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, Style.RESET_ALL)
        message = super().format(record)
        return f"{log_color}{message}{Style.RESET_ALL}"

def setup_logger(name):
    """Sets up the custom logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Adding handler to logger
    logger.addHandler(ch)
    return logger
