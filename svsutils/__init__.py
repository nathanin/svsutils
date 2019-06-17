# __name__ = 'svsutils'
from .iterators import *
from .svs_reader import *
from .postprocess import *

import os
def repext(src, newext):
  """
  Make it deal with trailing / leading periods.
  """
  newtxt = '{}'.format(newext)
  return os.path.splitext(src)[0] + newtxt

__all__ = [
  # Utilities (here)
  'repext', 

  # Iterators
  'PythonIterator',
  'TensorflowIterator',

  # svs_reader
    'Slide',
    'reinhard'

  # postprocessing tools
    'define_colormap',
    'overlay_img'
]