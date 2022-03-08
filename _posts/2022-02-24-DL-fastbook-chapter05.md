```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.all import *
import pandas as pd
```

    [K     |████████████████████████████████| 720 kB 3.2 MB/s 
    [K     |████████████████████████████████| 1.2 MB 51.2 MB/s 
    [K     |████████████████████████████████| 48 kB 5.6 MB/s 
    [K     |████████████████████████████████| 189 kB 60.7 MB/s 
    [K     |████████████████████████████████| 55 kB 4.6 MB/s 
    [K     |████████████████████████████████| 51 kB 117 kB/s 
    [K     |████████████████████████████████| 558 kB 38.2 MB/s 
    [K     |████████████████████████████████| 130 kB 59.2 MB/s 
    [?25hMounted at /content/gdrive
    

#Image Clssification
This chapter covers all aspects of the deep learning application in image classification, such as the various types of layers, regularization methods, optimizers, how to combine layers into architecture, labeling techniques, and so on. It addresses the problem of determining what breed of pet is depicted in each image in the dataset, which indicates a multi-category target.
##Labeling Data
This section attempts to find out how the breed name of each pet can be extracted from each image. There are two common ways of providing data as:
*   Data items are provided in files located in seperate folders or with filenames representing information about those items.
*   Data items that may include filenames are organized into the rows in a table, e.g., in CSV format, as well as a link to associated data in other formats such as images.

Underneath, we firstly browse the contents of the data folder and then offer a way to extract the label from filename:



```python
# setting the path to data location
path = untar_data(URLs.PETS)

# list of data folder contents
path.ls()
```
