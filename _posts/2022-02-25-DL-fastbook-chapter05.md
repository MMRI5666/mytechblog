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
     
As we can see, the file names include pets breeds' name, and we, hence, are able extract them using *regex* as shown below:
```python
# example of how to extract the pet breed name from file name
fname = (path/'images').ls()[0]
re.findall(r'(.+)_\d+.jpg$', fname.name)
```
     ['basset_hound']
     
Now that we know the location of the data and found the way to extract the label, we use the a special API offered by fastai, named `DataBlock` to construct the data block. The labels can be passed to this API using the class `RegexLabeller`.
```python
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items = get_image_files,
                 splitter = RandomSplitter(seed=41),
                 get_y = using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = pets.dataloaders(path/'images')
```
Two of the most important parameters used to construct the data block above are as:
*   `item_tfms=Resize(460)', and 
*   `batch_tfms=transforms(size=224, min_scale=0.75),
which lead to next important topic that is Presizing.

## Presizing
*Presizing* is an image augmentation strategy offered by `fastai`, which aims to minimize data destruction while maintaining good performance.<br>
Presizing adopts two strategies:
1.   Resize images to relatively "large" dimensions
2.   Compose all of the common augmentation operations (including a resize to the fianl target size) into one, and perform them on the GPU only once at the end of processing.

Presizing strategy performs the two steps shown by   , as described below:
1.   *Crop full width or height*: is in `item_tfms` and applied individually to each image before being copied to GPU. While the crop area on the training set is randomly taken out, the center square of the image is chosen on the validation set.
2.   *Random crop and augment*: is in `batch_tfms`, and applied to a batch all at once on the GPU. On the training set, the random crop and any other augmentations are done first. Whereas, the resize to the final size needed for the model is done on the validation set.

Figure ..., shows a comparison of fastai's data augmentation strategy (left) and the traditional  approach (right).

## Checking and Debugging a DataBlock
You can verify the dataset by using the follwoing two code scripts:
```python
dls.show_batch(nrows=2, ncols=2)
```

![image_01](/mytechblog/images/2022-02-25-DL-fastbook-chapter05/image_01.png)

```python
pets.summary(path/'images')
```
The method `summary` of the `DataBlock` object shows detail description about the status of the data inside, as well as any warning. To see an example of that refer to the page 192 of the the book.

Now, let's learn the model:
```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(3)
```
     Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
     100%
     83.3M/83.3M [00:03<00:00, 53.7MB/s]
     epoch	train_loss	valid_loss	error_rate	time
     0	1.478976	0.342973	0.108254	01:19
     epoch	train_loss	valid_loss	error_rate	time
     0	0.479528	0.331214	0.106901	01:21
     1	0.363403	0.241632	0.086604	01:25
     2	0.215266	0.191870	0.066982	01:23

Surprisingly, the table above provides some values for training and validation loss. As we remember, the loss is the result of evaluating loss function, which we determine in the model to optimize the parameters. Even though no loss function has been provided, how has FastAI calculated the loss for training and validation? In fact, depending on the type of data and the model, *fastai* chooses an appropriate loss function. In this cast of image classification, which uses image data and produces a categorical outcome, *fastai, by default, selects *cross-entropy loss*.

## Cross-Entropy Loss
A cross-entropy loss is similar to the loss function, which was developed using a sigmoid function, but has two privileges over it:
1.   It works effectively with models that have multi-categorical dependent variable.
2.   Leads to faster and more reliable training.
To unlock cross-entropy loss functionality, we need to know more about actual data (lables) and activations.

## Viewing Activations and Labels
To look at the activations layer of the model, let's select one batch of real data from the `DataLoader`: 

```python
x, y = dls.one_batch()
len(y), y
```
     (64,
      TensorCategory([35, 10, 32,  2, 35, 35, 22, 18, 17, 24, 28, 36, 26,  8, 31, 16, 30, 23, 30, 22, 25, 23, 17, 25, 29,  3, 16,  4, 24, 11, 33, 11, 18,  3, 15, 22,  1, 32,  1, 18, 17, 22,  2, 32, 11,  3,  4, 14,
               1, 24, 32, 17,  5, 19,  5, 35, 34, 31, 23, 12,  6, 18, 36,  9], device='cuda:0'))

The batch size is 64, i.e., it has 64 rows, each row contains a single integer between 0 and 36 that represent 37 possible pet breeds. The predictions, i.e., activations of the final layer of the neural network, can be retrieved using the method `Learner.get_preds`. This function takes either a dataset index (0 for train and 1 for valid) or an iterator of batches as the parameter. So, we simply pass a list of our sample batch to receive the predictions. We can expect it to return the predictions along with targets as shown below:

```python
preds, targets = learn.get_preds(dl=[(x, y)])
```

But since we already have the targets, we can ignore it by assign it to a special variable _:



><pre  style='font-style: normal; white-space: pre-wrap; background-color: #F2F2F2; border-bottom-color: #F2F2F2; color: #1A1A1A'>
In PyTorch, <em>nll_loss</em> does not take logarithm, however, it assumes that the log of <em>softmax</em> ouput was already taken. TyTorch has another function called <em>log_softmax</em> that combines log and softmax in a fast and accurate way, and <em>nll_loss</em> is designed to be used after log_softmax.</pre>
