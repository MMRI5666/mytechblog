# Image Classification


Here's the table of contents:

1. TOC
{:toc}

This chapter covers all aspects of the deep learning application in image classification, such as the various types of layers, regularization methods, optimizers, how to combine layers into architecture, labeling techniques, and so on. It addresses the problem of determining what breed of pet is depicted in each image in the dataset, which indicates a multi-category target.

## Labeling Data
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

```python
# setting the path to data location
path = untar_data(URLs.PETS)

# list of data folder contents
path.ls()
```
     
 100.00% [811712512/811706944 00:17<00:00]
(#2) [Path('/root/.fastai/data/oxford-iiit-pet/annotations'),Path('/root/.fastai/data/oxford-iiit-pet/images')]

The *annotations* directory contains information about where the pets are, which is not our concern. Therefore, we have to dig into the *images* directory.

```python
(path/"images").ls()
```
     (#7393) [Path('/root/.fastai/data/oxford-iiit-pet/images/basset_hound_96.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/Abyssinian_116.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/shiba_inu_35.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/basset_hound_115.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/shiba_inu_191.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/Sphynx_2.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/miniature_pinscher_141.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/english_cocker_spaniel_176.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/leonberger_111.jpg'),Path('/root/.fastai/data/oxford-iiit-pet/images/newfoundland_60.jpg')...]
     
