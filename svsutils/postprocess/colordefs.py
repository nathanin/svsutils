import numpy as np

"""
Colors from playing around in

http://vrl.cs.brown.edu/color
"""

builtin_cmaps = {
  'elf': np.array(
    [
     (130,84,45),   # brown
     (214,7,36),    # red
     (37,131,135),   # turquois
     (244,202,203),  # pink
     ]
   ),
  'elfr': np.array(
    [
     (244,202,203),
     (37,131,135),   # turquois
     (130,84,45),   # brown
     (214,7,36),    # red
     ]
   ),
  'elf5': np.array(
    [(130,84,45),   # brown
     (222,16,16),    # red
     (9,22,203),  # green
     (37,131,135),   # turquois
     (244,202,203)]  # pink
   ),
  'greens': np.array(
    [(32,142,183), 
     (141,210,216), 
     (25,79,70), 
     (92,221,159), 
     (29,138,32), 
     (150,218,49), 
     (127,136,97)]
   ),
  'pinks': np.array(
    [(213,152,199), 
     (250,126,227), 
     (254,41,226), 
     (185,185,185), 
     (253,89,37), 
     (254,162,122), 
     (250,85,122)]
   ),
  'blues': np.array(
    [(32,142,183), 
     (133,229,221), 
     (14,80,62), 
     (64,225,140), 
     (25,71,125), 
     (175,198,254), 
     (37,128,254)]),
  'ten': np.array(
    [(158,115,184), 
     (89,141,131), 
     (151,68,72), 
     (39,76,86), 
     (248,47,101), 
     (209,28,197), 
     (114,18,255), 
     (243,66,7), 
     (96,36,158), 
     (4,138,209)]),
}

def define_colors(name, n_colors, add_white=False, shuffle=False):
  try:
    colors = builtin_cmaps[name]
  except:
    raise Exception('Custom colormaps not yet. \
      Try one of : {}'.format(list(builtin_cmaps.keys()))
    )

  if add_white:
    colors = np.concatenate([colors, [[255]*3]])

  if shuffle:
    np.random.shuffle(colors)

  return colors