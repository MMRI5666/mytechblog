# Image Classification


Here's the table of contents:

1. TOC
{:toc}

This chapter covers all aspects of the deep learning application in image classification, such as the various types of layers, regularization methods, optimizers, how to combine layers into architecture, labeling techniques, and so on. It addresses the problem of determining what breed of pet is depicted in each image in the dataset, which indicates a multi-category target.

##Labeling Data
This section attempts to find out how the breed name of each pet can be extracted from each image. There are two common ways of providing data as:
*   Data items are provided in files located in seperate folders or with filenames representing information about those items.
*   Data items that may include filenames are organized into the rows in a table, e.g., in CSV format, as well as a link to associated data in other formats such as images.

Underneath, we firstly browse the contents of the data folder and then offer a way to extract the label from filename:
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

In a computer, everything is represented as a number, therefore, to view the numbers that make up this image, we have to convert it to a `NumPy` array or `PyTorch` tensor. The following contents introduce some tricks in working with these two data structures, but if you want to know more about them you can refer to the following links:
- [NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
- [PyTorch tensors](https://pytorch.org/docs/stable/tensors.html)
