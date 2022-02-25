### Metrics Computation Using Broadcasting
*Metric* is a numeric measure that enables us to evaluate our model's performance. It is calculated based on the number of labels in our dataset that are predicted by the model.
Taking the average of values calculated using each of the loss functions explained in the previous section, i.e. MEA and RSME, over the whole dataset can be used as a metric for a model. However, using the model's *accuracy* as the metric for classification models is more common since neither MAE nor RSME seems very easy to understand to most people.<br>
Metrics are calculated over the validation set to prevent the model from overfitting. Model overfitting may not be a risk the 3-or-7 basic classifier model explained in the previous section because it does not adopt any training component. Conversely, overfitting poses a major concern in training every model in machine learning and deep learning.<br>
To calculate the accuracy of our simple classifier model, we need to instantiate tensors for 3s and 7s that are taken out form validation set. In MNIST dataset, the validation set is placed on a seperate directory named `valid`. So, the scripts for creating the validation tensors will be as shown below:


```python
# validation tensor for 3s samples from validation dataset
valid_3_tens = torch.stack([tensor(Image.open(img)) 
                          for img in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float() / 255

# validation tensor for 7s samples from validation dataset
valid_7_tens = torch.stack([tensor(Image.open(img)) 
                          for img in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens / 255

#checking the validation tensors' shape
valid_3_tens.shape, valid_7_tens.shape
```
  (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))



To calculate a metric for overall model accuracy in detecting 3, we will firstly need to define a function that calculate the distance for every image in the validation set. When applying this function to validation tensor, the distance from ideal 3 will be calculated for every single image using tensor's *broadcasting* feature as shown below:


```python
def mnist_dist(a, b):
  return (a-b).abs().mean((-1, -2))

valid_3_dist = mnist_dist(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
```
  (tensor([0.1263, 0.1413, 0.1430,  ..., 0.1332, 0.1471, 0.1469]),
 torch.Size([1010]))



In broadcasting, PyTorch automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank. There are two worth-mentioning point about broadcasting as below:
*   Expanding the lower-randed tensor does not mean that PyTorch copies mean3 1010 times or allocate any additional memory to do this, it actually pretends to have a tensor of that shpape.
*   PyTorch performs the calculation in C (or CUDA if we use GPU), and this the secret of thousands of times lower computation time.
<br>Now we need a function to verify whether or not each of the images is 3 by comparing its distrance from ideal 3 and ideal 7.


```python
def is_3(x):
  return mnist_dist(x, mean3) < mnist_dist(x, mean7)

# correctness percentage of predicting 3 
accuracy_3s = is_3(valid_3_tens).float().mean()

# correctness percentage of detecting 7 (non 3 image).
accuracy_7s = (1 - is_3(valid_7_tens).float().mean())

accuracy_3s, accuracy_7s, (accuracy_3s + accuracy_7s) / 2
```
  (tensor(0.9168), tensor(0.9854), tensor(0.9511))





The result above shows an accuracy over 90% on both predicting correctly 3s and correctly detecting 7s as not beeing a 3, which is quite accetable for such simple classifier model.
