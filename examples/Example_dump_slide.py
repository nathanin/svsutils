from svsutils import Slide, PythonIterator
from svsutils import repext
from svsutils import cpramdisk

import traceback
import argparse
import cv2
import os

def compute_fn(slide, args):
  print('Slide with {} tiles'.format(len(slide.tile_list)))

  it_factory = PythonIterator(slide, args)
  for k, (img, idx) in enumerate(it_factory.yield_one()):
    if k % 1000 == 0:
      print('Batch {:04d}: {}'.format(k, img.shape))
    slide.place(img[:,:,::-1], idx, 'wsi', 
                mode='full', 
                clobber=True)
  ret = slide.output_imgs['wsi']
  return ret





def main(args):
  with open(args.lst, 'r') as f:
    srclist = [x.strip() for x in f]

  for src in srclist:

    dst = repext(src, args.suffix)
    if os.path.exists(dst):
      print('Exists', src, '-->', dst)
      continue

    # Loading data from ramdisk incurs a one-time copy cost
    rdsrc = cpramdisk(src, args.ramdisk)
    print('File:', rdsrc)

    try:
      slide = Slide(src, args)
      slide.initialize_output('wsi', 3, mode='full', 
                              compute_fn=compute_fn)
      ret = slide.compute('wsi', args)
      print('Saving {} --> {}'.format(ret.shape, dst))
      cv2.imwrite(dst, ret)
    except Exception as e:
      traceback.print_tb(e.__traceback__)
    finally:
      print('Removing {}'.format(rdsrc))
      os.remove(rdsrc)





if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('lst')
  p.add_argument('--suffix', default='.img.jpg') # make sure it starts with a `.`

  p.add_argument('-b', dest='batchsize', default=1, type=int)
  p.add_argument('-r', dest='ramdisk', default='/dev/shm', type=str)

  # Slide options
  p.add_argument('--mag',   dest='process_mag', default=5, type=int)
  p.add_argument('--chunk', dest='process_size', default=256, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.05, type=float)

  args = p.parse_args()
  main(args)
