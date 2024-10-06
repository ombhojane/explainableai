# explainableai/__init__.py
from .core import XAIWrapper

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
__all__ = ['XAIWrapper']