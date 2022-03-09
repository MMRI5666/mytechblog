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
```python
preds, _ = learn.get_preds(dl=[(x, y)])
preds[0]
```
     TensorBase([7.3856e-09, 6.4286e-11, 1.2409e-09, 2.7335e-11, 4.6427e-09, 8.0484e-09, 2.9370e-09, 1.5099e-08, 5.6245e-09, 1.2676e-08, 9.6034e-10, 5.8992e-09, 1.8227e-09, 2.5063e-10, 3.5726e-10, 2.9001e-10,
        6.3725e-09, 1.6889e-10, 1.5992e-10, 1.5381e-10, 4.3895e-10, 3.6065e-10, 9.9710e-08, 2.9064e-10, 7.2608e-11, 2.6796e-09, 4.1463e-11, 5.0358e-10, 2.8368e-10, 2.2414e-10, 3.2216e-10, 2.4286e-09,
        3.3916e-06, 9.0170e-09, 2.1684e-10, 1.0000e+00, 7.1604e-09])
        
In fact, the actual predictions are 37 probabilities between 0 and 1, which sum up to 1 in total:
```python
len(preds[0]), preds[0].sum()
```

Behind the scene, an activation function called *softmax* is used to transform the activation of the model into predictions.

## Softmax
Softmax, similar to sigmoid function, is applied to final layer of trained model with multi-categorical dependent variable to ensure that the activations are all between 0 and 1, and that they sum to 1.
```python
def softmax(x):
  return exp(x) / exp(x).sum(dim=1, keepdim=True)
```

The softmax function is also available in PyTorch just like the sigmoid function. We will check now if softmax returns the same values as sigmoid in the first column, and if those values are subtracted from 1 in the second column.
```python
sm_acts = torch.softmax(acts, dim=1)
sm_acts
```
Some other functions could be developed that can produce activations between 0 and 1, and sum to 1, however, no other function can behave smoothly and symmetrically in similar ways to sigmoid as softmax does. Additionally, the softmax integrated collaborates with the loss function.<br>
><pre  style='font-style: normal; white-space: pre-wrap; background-color: #FFF5E6; border-color: #CCCCCC; color: #404040'>
<b>Important</b><br>In applying <em>softmax</em>, if one of an existed number in the activations is slightly bigger than the others, applying the exponential amplify that, resulting being closer to 1 in softmax outcome. Therefore, increasing the activation of the correct column as much as possible entails reducing the activation of the other columns.</pre>

In PyTorch, you can find a function called *nll_loss*, which does the same thing as *sm_acts[range(n), targ], except that it takes negative values. Possibly you are wondering why this function needs to accept negatives! This is because when applying a logarithm, we may get negative values. However, the next question would be why we need to take the logarithm.
```python
-sm_acts[idx, targ]
```

```python
F.nll_loss(sm_acts, targ, reduction='none')
```

<strong>Why taking the logarithm is useful?</strong><br>
Although the loss function, which was developed in the previous section appears to work quite well, it has a major drawback. When using probabilities, which are varying exclusively between 0 and 1, the model will not care whether it predicts 0.99 or 0.999. In fact, these numbers are so close to each other, however in another sense, 0.999 is ten times more confident than 0.99. Transforming the numbers between 0 and 1 to instead be between negative infinity and infinity seems to alleviate effectively this deficiency. The mathematical function, i.e., *logarithm*, which is available in PyTorch as `torch.log`, can take care of this transformation. 
><pre  style='font-style: normal; white-space: pre-wrap; background-color: #E6EEFF; border-color: #CCCCCC; color: #404040'>
<b>Note</b><br>Mathematically, the <strong>logarithm</strong> function is:
&emsp;&emsp; y = b ** a
&emsp;&emsp; a = log(y, b)             assuming that log(y, b) returns *log y base b.
In Python, log uses the special number e(2.718...) as the base.</pre>

><pre  style='font-style: normal; white-space: pre-wrap; background-color: #FFF5E6; border-color: #CCCCCC; color: #404040'>
<b>Important</b><br>The following equation is a one of teh key mathematical expression in deep learning:
&emsp;&emsp; log(a * b) = log(a) + log(b)</pre>

The importance of the equation mentioned above is unveiled when we note that while the underlying signal increases exponentially, the logarithms increase linearly. Computer scientists are very interested in using this concept to replace arithmetic operations that produce extremely large and extremely small numbers with operations that are much easier for computers to handle in terms of both computation and memory loads.<br>
Now, we know that why we use logarithms in deep learning and how taking the mean of the positive or negative log of the probabilities (depending on whether it's the correct or <em>incorrect</em> class) returns the *negative log likelihood loss*.

><pre  style='font-style: normal; white-space: pre-wrap; background-color: #F2F2F2; border-bottom-color: #F2F2F2; color: #404040'>
In PyTorch, <em>nll_loss</em> does not take logarithm, however, it assumes that the log of <em>softmax</em> ouput was already taken. TyTorch has another function called <em>log_softmax</em> that combines log and softmax in a fast and accurate way, and <em>nll_loss</em> is designed to be used after log_softmax.</pre>


><pre  style='font-style: normal; white-space: pre-wrap; background-color: #E0EBEB; border-color: #CCCCCC; color: #404040'>
<b>Definition</b><br>The composition of applying softmax, and then taking the log likelihood of that is called <em>cross-entropy loss</em>, and is available in PyTorch as nn.CrossEntropLoss (which practically does <em>log_softmax</em> and then <em>nll_loss</em><br>
&emsp;&emsp;<b>loss_func = nn.CrossEntropyLoss(),/b>

