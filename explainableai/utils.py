# utils.py

# Import colorama and its components
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Example utility function using colorama for colored logs
def log_data_processing_step(step_description):
    logger.info(f"{Fore.BLUE}{step_description}{Style.RESET_ALL}")

# Example utility class
class DataProcessor:
    def process_data(self, data):
        logger.info(f"{Fore.YELLOW}Starting data processing...{Style.RESET_ALL}")
        # Implement data processing logic here
        logger.info(f"{Fore.YELLOW}Data processing completed.{Style.RESET_ALL}")

# Add your actual utility functions and classes below
# Ensure that any function or class using Fore or Style includes the imports above

def some_utility_function():
    # Example function using Fore and Style
    logger.info(f"{Fore.GREEN}This is a green message.{Style.RESET_ALL}")
    # Rest of the function...

class SomeUtilityClass:
    def example_method(self):
        logger.info(f"{Fore.RED}This is a red message.{Style.RESET_ALL}")
        # Rest of the method...


