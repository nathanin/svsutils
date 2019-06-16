# __name__ = 'svsutils'
from .iterators import *
from .svs_reader import *

__all__ = [
  'PythonIterator',
  'TensorflowIterator',

  # svs_reader
    "Slide",
    "reinhard"
]