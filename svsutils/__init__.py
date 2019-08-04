# __name__ = 'svsutils'
from .iterators import *
from .svs_reader import *
from .postprocess import *

import os
import shutil 
def repext(src, newext):
  """
  Make it deal with trailing / leading periods.
  """
  newtxt = '{}'.format(newext)
  return os.path.splitext(src)[0] + newtxt

## TODO This can be a context with syntax like:
## with cpramdisk(src, dst) as slide:
##     ... stuff
## 
## A context would eliminate the try-except-finally pattern
## inside of the frontend script.
def cpramdisk(src, dst):
  src_base = os.path.basename(src)
  dst_full = os.path.join(dst, src_base)

  # Somehow check the contents of src and dst for integrity
  if os.path.exists(dst_full):
    print('Warning: cpramdisk found the target already exists:')
    print('{}'.format(dst_full))
  else:
    shutil.copyfile(src, dst_full)
      
  return dst_full

__all__ = [
  # Utilities (here)
  'repext', 
  'cpramdisk'

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