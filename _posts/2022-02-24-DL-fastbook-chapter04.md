# This is chapter 4 of Deep Learning for Coders with fastai & PyTorch

Here's the table of contents:

1. TOC
{:toc}

## Computer vision begins with `## Pixels`
The first step in understanding what is going on inside a computer vision model is to comprehend how computer interact with images. This chapter tries to explain the foundation of computer vision conducting some experimentations on the very popular data set MNIST, which contains images of handwritten digits collected by the National Institute of Standards and Technology by *Yann Lecun* and his Colleagues. Therefore, we need to download the sample dataset, but before that we should know how install the `fastbook` contents, and how to import `fastai` library.

```python
# installing fastbook contents and import it
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```

     |████████████████████████████████| 720 kB 8.9 MB/s 
     |████████████████████████████████| 189 kB 24.3 MB/s 
     |████████████████████████████████| 48 kB 5.8 MB/s 
     |████████████████████████████████| 1.2 MB 36.9 MB/s 
     |████████████████████████████████| 55 kB 4.0 MB/s 
     |████████████████████████████████| 51 kB 380 kB/s 
     |████████████████████████████████| 558 kB 55.2 MB/s 
     |████████████████████████████████| 130 kB 56.7 MB/s 
Mounted at /content/gdrive
```python
# import libraries: fastbook, fastai, and pandas
from fastbook import *
from fastai.vision.all import *
import pandas as pd
```

In a computer, everything is represented as a number, therefor, to view the numbers that make up this image, we have to convert it to NumPy or PyTorch tensor.

