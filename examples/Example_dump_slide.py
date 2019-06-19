from svsutils import Slide, PythonIterator
import argparse
import cv2
import os

def compute_fn(slide, args):
  it_factory = PythonIterator(slide, args)
  for k, (img, idx) in enumerate(it_factory.yield_one()):
    slide.place(img[:,:,::-1], idx, 'wsi', 
                mode='full', 
                clobber=True)
    if k % 50 == 0:
      print(k)

  ret = slide.output_imgs['wsi']
  return ret

def main(args):
  with open(args.lst, 'r') as f:
    srclist = [x.strip() for x in f]

  for src in srclist:
    dst = os.path.splitext(src)[0] + '.{}'.format(args.t)
    if os.path.exists(dst):
      print('Exists', src, '-->', dst)
      continue
    slide = Slide(src, args)
    slide.initialize_output('wsi', 3, mode='full', 
                            compute_fn=compute_fn)
    ret = slide.compute('wsi', args)
    cv2.imwrite(dst, ret)
  

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('lst')
  p.add_argument('-t', default='img.jpg')
  p.add_argument('-b', dest='batchsize', default=1, type=int)
  p.add_argument('-r', dest='ramdisk', default='./', type=str)

  # Slide options
  p.add_argument('--mag',   dest='process_mag', default=5, type=int)
  p.add_argument('--chunk', dest='process_size', default=512, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.5, type=float)

  args = p.parse_args()
  main(args)
