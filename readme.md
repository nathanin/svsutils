## SVSUTILS

An object for interfacing __usually__ Aperio's SVS slide images. Wraps OpenSlide.

Find foreground, tile foreground, yield tiles. See tests for some example scripts.

### Installation
Prerequisites:
```
openslide-python
opencv
numpy

# optional
tensorflow
MulticoreTSNE
umap-learn
```

To install:
```
/usr/env/bin/pip install -e .
```

To use:
```
$ python
>>> from svsutils import Slide
```


