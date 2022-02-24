# This is chapter 4 of Deep Learning for Coders with fastai \& PyTorch

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

In a computer, everything is represented as a number, therefore, to view the numbers that make up this image, we have to convert it to a `NumPy` array or `PyTorch` tensor. The following contents introduce some tricks in working with these two data structures, but if you want to know more about them you can refer to the following links:

- [NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
- [PyTorch tensors](https://pytorch.org/docs/stable/tensors.html)

## Basic Image Classifier Model
This section of the book comes up the idea of using the pixel similarity as the very basic method to classify images. For this purpose, the average pixel value for every pixel of the 3s samples, and the for 7s samples are calculated. As a result, we have two arrays/tensors containing the pixel values for two images that we might call the "ideal" 3 and 7. Hence, to classify an image as 3 or 7, we can evaluate which of these two ideal digits the image is similar to.

### Constructing the base model
**Step 1**: Calculating the average of pixel values of each of two sample groups of 3s and 7s. Creating a tensor containing all of our 3s stacked together. For this, Python list comprehension is used to create a plain list of the single image tensors.
```python
# creating a tensor containing all of 3s sample images stacked together 
# using list comprehension
three_tensors= [tensor(Image.open(img)) for img in threes]

# and the same for 7s sample images
seven_tensors = [tensor(Image.open(img)) for img in sevens]

# checking the number of items in each tensor
len(three_tensors), len(seven_tensors)
```
     (6131, 6265)

**Step 2**: Stacking up all the image tensors in this list into a single three-dimensional tensor (rank-3 tensor) using PyTorch stack function. The values stored in stacked tensor is casted to float data types, as required by some PyTorch operations, such as taking a mean.
```python
# stacking up all the image tensors in the list in to one rand-3 tensor, and 
# cast it to float types.
stacked_threes = torch.stack(three_tensors).float() / 255
stacked_sevens = torch.stack(seven_tensors).float() / 255

# checking the stacked tensor's size
stacked_threes.shape
```
     torch.Size([6131, 28, 28])

{% include info.html text="<strong>Definition</strong>:
<br>Tensor's rank is the number of axes or dimensions in a tensor, shape is the size of each axis of a tensor." %}
```python
# rank or dimension of a tensor using len()
len(stacked_threes.shape)

# or directly using ndim
stacked_threes.ndim
```
     3

```python
stacked_threes.shape[1]
```
     28

**Step 3**: Computing what the ideal 3 and 7 look like through calculating the mean of all the image tensors by calling mean for every index position over images along dimension 0 of both stacked rank-3 tensor.
```python
# calculating mean for index position over the images along dimension 0 
# of the both rank-0 stacked tensors of 3s and 7s
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)

#showing ideal 3
show_image(mean3)
```
![](/images/2022-02-24-DL-fastbook-chapter04/ideal_3.png "ideal 3")

### Defining Loss Function
Now that the 3-or-7 classifier model is ready to use, we can pick up an arbitrary 3 and calculate its distance from the "ideal digits". Two main alternatives to measure distance in this case are as:
* MAE: the Mean absolute value of difference, which is also called L1 norm.

```pyrhon
# calcuting the distance from ideal 3 using MEA and RSME
dist_3_mea = (a_3 - mean3).abs().mean()
dist_3_rsme = ((a_3 - mean3) ** 2).mean().sqrt()
dist_3_mea, dist_3_rsme
```
     (tensor(0.1114), tensor(0.2021))
