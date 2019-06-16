"""

June 2019
"""

import numpy as np
from svsutils.svs_reader import Slide

try:
  import tensorflow as tf
except:
  print('Failed to import tensorflow')


class PythonIterator():
  """
  A class for building an iterator for use like:

  it_factory = PythonIterator(..)

  for img, idx in it_factory.yield_one(...):
    ...

  Parameters:
  src (svsutils.Slide): Path to slide (required)
  img_idx (bool): Whether to yield a tuple (img, index). (True)
  batchsize (int): Size of batches to yield. 
                   Batches will be (batchsize, h, w, c). (1)

  """
  def __init__(self, slide, args, **kwargs):
    self.arg_defaults = {
      'batchsize':1,
      'img_idx':True
    }
    # Check identity of slide
    # assert 

    self.slide = slide
    self.extract_args(args)
    assert self.batchsize > 0

  def compute_output(self, outname, **kwargs):
    """
    Run the output function
    """
    fn = self.compute_fns[outname]
    ret = fn(**kwargs)
    return ret

  def extract_args(self, aparse_space):
    # Extract parts of aparse_space that are 
    # 1:1 in the dictionary
    input_keys = list(aparse_space.__dict__.keys())
    for key, val in self.arg_defaults.items():
      if key in input_keys:
        setattr(self, key, aparse_space.__dict__[key])
      else:
        setattr(self, key, val)
    
  def yield_one(self, shuffle=True):
    """
    for img, idx in slideobj.yield_one():
      ...

    """
    for idx in self.slide.generate_index(shuffle=shuffle):
      coords = self.slide.tile_list[idx]
      img = self.slide._read_tile(coords)
      yield img, idx

  def yeild_batch(self, shuffle=True):
    """
    it's faster to call yield_one() if batchsize=1
    """
    # np.array_split or while < batch size ?
    single_generator = self.yield_one(shuffle=shuffle)
    while True:
      bimg, bidx = [], []
      try:
        while n < self.batch_size:
          bimg.append(bimg)
          bidx.append(bidx)
          n+=1
      except:
        break
      finally:
        bimg = np.stack(bimg, axis=0)
        bidx = np.concatenate(bimg)
        yield bimg, bidx


class TensorflowIterator(PythonIterator):
  """
  A class for building an iterator for use like:

  my_iterator = 

  Parameters:
  src (svsutils.Slide): Path to slide (required)
  img_idx (bool): Whether to yield a tuple (img, index). (True)
  batchsize (int): Size of batches to yield. 
                   Batches will be (batchsize, h, w, c). (1)

  """
  def __init__(self, slide, batchsize=1, img_idx=True, prefetch=256,
               workers=6):
    super(TensorflowIterator, self).__init__(slide, batchsize=batchsize,
      img_idx=img_idx)
    assert 'tf' in dir() # How to check tensorflow is imported

    self.prefetch = prefetch
    self.workers = workers
    self.tfiterator = self.make_iterator()
  
  def wrapped_fn(idx):
    coords = self.slide.tile_list[idx]
    img = self.slide._read_tile(coords)
    return img, idx

  def read_region_at_index(self, idx):
    return tf.py_func(func = self.wrapped_fn,
                      inp  = [idx],
                      Tout = [tf.float32, tf.int64],
                      stateful = False)

  def make_iterator(self):
    # TODO allow sequential reading ?
    ds = tf.data.Dataset.from_generator(generator=self.slide.generate_index,
        output_types=tf.int64)
    ds = ds.map(read_region_at_index, num_parallel_calls=self.workers)
    ds = ds.prefetch(self.prefetch)
    ds = ds.batch(self.batchsize)

    if tf.executing_eagerly():
      return tf.contrib.eager.Iterator(dataset)
    else:
      return ds.make_one_shot_iterator()

    