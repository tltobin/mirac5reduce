# Set easily accessible version 
from importlib.metadata import version
__version__ = version(__name__)

# Imports packages contained in this directory
from . import cal, utils, reduce

