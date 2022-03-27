# Gradient Descent


Here's the table of contents:

1. TOC
{:toc}

## Overview
<p><em>Gradient descent</em> is by far the most crucial mathematical concept used in deep learning algorithms, and it is fair to say that deep learning would not be possible without it. The gradient vector can be interpreted as the "direction and rate of fastest increase". If the gradient of a function is non-zero at a point <em>p</em>, the direction of the gradient is the direction in which function increases most quickly from <em>p</em>, and the magnitude of the gradient is the rate of increase in that direction, the greatest absolute directional derivate. Furthermore, the gradient is the zero vector at a point if and only if it is a stationary point, where the <em>derivative</em> vanishes (<a href="https://en.wikipedia.org/wiki/Gradient">source</a>). From the calculus we know that a minimum of a function (perhaps a local minimum) happens in the point in which the function's derivative is equal to zero (as the first order necessary condition). This is the reason that gradient plays a fundamental role in optimization theory, where it is used to minimize or maximize a function by gradient descent or gradient ascent respectively. As a result, we now understand why gradient descent is essential to minimizing the loss of a deep learning model.</p>

The gradient of a function is closely related to its derivative. Basically, the gradient of a function <em>f</em> is the dual to its <em>total derivative</em>, or in other words, they are related in that the <em>dot product</em> of the gradient of <em>f</em> at the point <em>p</em> with another tangent vector <em>v</em> equals the <em>directional derivative</em> of <em>f</em>, which is depicted as below:
  
![gradient_derivative_01.svg](/mytechblog/images/2022-03-11-DL_gradient_descent/gradient_derivative_01.svg)


## Impact of <em>local minimums</em> on deep learning algorithms
<p>Local minima are always a risk in most optimizing iterative search algorithms, and gradient descent is no exception. Gradient descent undoubtedly guides the minizing loss in deep learning algorithms to the minima, but there is no guarantee to find a global minimum. We are now interested in understanding how being stuck in a local minima can affect deep learning algorithms.</p> <p>Deep learning practitioners claim that being trapped by a local minimum does not compromise deep learning algorithms since it searches for minimum in a high-dimensional space. A local minimum is a point that is the minimum of all dimensions, and this is very rare in deep learning algorithms. This is the magic behind the huge success of deep learning algorithms: high-dimensionality and complexity. This fact is very hard to digest, due to the impossibility of visualizing such a high-dimensional space. Another aspect of a deep learning algorithm that protects it against slipping into a local minimum is its wide diversity of weight parameters. Indeed, the diversity of weight configurations offers many acceptable local minimums, i.e., many good solutions, and drastically reduces the likelihood of local minima arising.</p>


## Gradient Descent in 1D
The following section shows, through an example, how to minimize a function by tracking its gradient and find a near minimum solution and the variable value at the minimum of the function. The funciton is as below:
<p><center>y = f(x) = 3x<sup>2</sup> + 3x + 4</center></p>
<p>
Two approaches are used in PyTorch and NumPy to solve the example. However, first, to illustrate the relationship between the function and its derivative, they are plotted with matplotlib. The codes are provided in each subsection respectively.
</p>
```python
# import libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
```

```python
# defining function
def f(x):
  return (3 * x ** 2 - 3 * x + 4)
```

### Plotting the function and its derivative
```python
# plot the function and its derivative
x = torch.linspace(-2, 2, 2000).requires_grad_()
y = f(x).sum()
y.backward()
xgrd = x.grad

plt.plot(x.data.detach().numpy(), f(x).detach().numpy(),
         x.data.detach().numpy(), xgrd.detach().numpy())

plt.grid()
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.legend(['y', 'dy'])
plt.show()
```
![gradient_derivative_01.svg](/mytechblog/images/2022-03-11-DL_gradient_descent/gradient_descent_0101.png)

### Finging solution using PyTorch
A randomly generated dataset for the variable is used to start the search for the near minimum value for the function and variable value. This proves that gradient-based minimizing has no effect on the initial solution and converges effectively to the minimum point. The minimizing, meanwhile, is not compromized by the local minimum in this case.

```python
# randomly generate the start and end points
edges = torch.randn(2)
print(f'start={edges.min()}, end={edges.max()}')

# randomly generate 100 points within the start and end
xt = ((edges.max() - edges.min()) * torch.rand(100) + \
    edges.min()).requires_grad_()
xt.data.min()
```
    start=0.021666323766112328, end=0.20076826214790344
    tensor(0.0295)

```python
# learning parameters
learn_rate = 0.01
train_epochs = 150

# run through training and store all the results
modelparams = np.zeros((train_epochs, 2))

# iteratively search for function's minimum
for i in range(train_epochs):
  yt = f(xt).sum()
  yt.backward()
  xt.data -= learn_rate * xt.grad.data
  modelparams[i, 0] = xt.data.min()
  modelparams[i, 1] = xt.grad.data.min()
  xt.grad = None
```

### Plotting the result
```python
# plot the results

x = torch.linspace(-2, 2, 2000).requires_grad_()
y = f(x).sum()
y.backward()
xgrd = x.grad

plt.plot(x.data.detach().numpy(), f(x).detach().numpy(),
         x.data.detach().numpy(), xgrd.detach().numpy())

plt.grid()
plt.xlabel('x')
plt.ylabel('y=f(x)')
plt.legend(['y', 'dy'])
plt.scatter(modelparams[train_epochs-1, 0], 
            f(modelparams[train_epochs-1, 0]), color='red')
plt.scatter(modelparams[train_epochs-1, 0], 
            modelparams[train_epochs-1, 1], color='brown')
plt.legend(['f(x)', 'df', 'f(x) min', 'df at min'])
plt.show()
```
![gradient_derivative_01.svg](/mytechblog/images/2022-03-11-DL_gradient_descent/gradient_descent_0102.png)

### Plotting local minimum and its correspoding dradient value over the iterations
```python
# plot the gradient over iterations
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

for i in range(2):
  ax[i].plot(modelparams[:, i], 'o-')
  ax[i].set_xlabel('Iteration')
  ax[i].set_title(f'Final estimated minimum: {modelparams[train_epochs-1, 0]:.5f}')

ax[0].set_ylabel('Local minimum')
ax[1].set_ylabel('Derivative')
plt.show()
```
![gradient_derivative_01.svg](/mytechblog/images/2022-03-11-DL_gradient_descent/gradient_descent_0103.png)

## Finding solution using NumPy

```python
# define derivative function of f(x)
def derivf(x):
  return 6 * x -3
```

```python
# random starting point
x = np.linspace(-2, 2, 2000)
localmin = np.random.choice(a=x, size=1)
print(localmin)

# learning parameters
learning_rate = 0.01
training_epochs = 175

# run through training and store all the results
modelparams = np.zeros((training_epochs, 2))
for i in range(training_epochs):
  grad = derivf(localmin)
  localmin = localmin - learning_rate * grad
  modelparams[i, :] = localmin, grad
```

Plotting the solution, as well as plotting the tracking of how the search iterations converge to the local minima and derivatives values are identical to the figures in the previous sub-section.

<br><br>

## Vanishing Gradient Problem
<p>When minimization of a function, as shown in Figure 4, begins at point <em>s</em>, the derivative's value is negative, and it moves in the opposite direction of the derivate at that point with a step size proportional to the magnitude of the derivate. As the minimization process progresses, the derivative becomes flatter, and each subsequent step size becomes smaller as a result. In practice, the deviate value appears to be approaching zero, causing the minimization process to run longer than the number of training epochs before reaching the local minima, i.e., point <em>p</em>. This phenmonen is called <em>vanishing gradient descent</em>.
</p>

![gradient_descent_0104_s.png](/mytechblog/images/2022-03-11-DL_gradient_descent/gradient_descent_0104_s.png "Figure-4, Vanishing gradient problem.")

<p>
Because the learning relies on the gradient's value in deep learning based on multiple hidden neural networks, the vanishing gradient descent occurrence prevents the model from learning effectively. The initial layers of this neural network are critical because these layers are where the neural network's building blocks are learned from the input dataset's simple patterns (features). The network's derivatives are the result of multiplying the preceding layers, layer by layer, from the last to the first in the backpropagation step of a NN model training process. This results in an exponential decrease in the derivative as it approaches zero, eventually disappearing.
</p>
<p>
  The impact of gradient vanishing on NN model training will be investigated in greater depth. To begin, we should bear in mind that the goal of backpropagation is to minimize the loss by adjusting the weights and biases across the neural network based on minimizing the loss. Propagating the error back to the layers very close to the input layer is taken into effect via backpropagation process. A partial derivative of the gradient is used to implement this propagation. This makes it easier to explain when the derivative is exponentially approaching zero, the adjustment of the model parameters, i.e., weights, biases, and propagation of the error to the initial layers, and then the model training will be disrupted. As a result, the neural network's initial layers are rendered useless since they are not able to learn anything.
</p>
<p>
It is important to understand why gradient descent tends to zero. As we know, neural networks traditionally use <em>sigmoid function</em> activation function within the layers. Figure 5, which shows the sigmoid function and its derivative, can interpreted in a way that the sigmoid function maps any input value, regardless of its magnitude, to a value within the range of o to 1. In fact, when it receives either a large or a small input value, it saturates the mapping values to the red zones on the right or left side reppectively, which their corresponding derivative values will be very close to zero. 
</p>
![gradient_descent_0107.png](/mytechblog/images/2022-03-11-DL_gradient_descent/gradient_descent_0107.png "Figure-5, Sigmoid function and its derivative.")

### Solution
There are some ways poroposed to alleviate the vanishing gradient descent in deep learning, which are outlined as followed:

1.  <b>Weight initialization</b><br>
Random weight initialization allows the model to be trained with different starting points that may be more conducive to learning. Choosing either too large or too small weights causes some of the non-linear activation functions, such as the sigmoid function, to return very small values, almost equal to zero. <em>Xavier initialization</em> is a way that selects the weights of the network from an intermediate range of values, while passing through the various layers, in such a way that the variance is not significant.

2.  <b>Using non-saturating activation functions</b><br>
The simplicity of computing the Sigmoid's derivative function is the reason for this function to be the most commonly used activation function. However, we have seen that the drawback to this function is a high exposure to the risk vanishing gradient, which is why we have classified it under saturated activation functions. On the other hand, there are some better alternatives such as <em>Rectified Linear Unit</em> (ReLU) do not saturate when receiving positive values, resulting in being more resilient to the vanishing gradient. However, it faces the problem of dying ReLUs, which means some neurons stop generating anything other than zeroes. This problem can be resolved by using Leaky ReLU instead of ReLU, which can address this issue.<br>
![gradient_descent_0109.png](/mytechblog/images/2022-03-11-DL_gradient_descent/gradient_descent_0109.png "Figure-6, activation functions.")
  4.  <b>Batch normalization</b><br>
A layer in the network is added right before the activation layer to transform all of the input data to a reasonable range of values and prevent them from being saturated by the activation function. This new layer is in charge of standardizing and normalizing the input from the previous layer before sending it to the activation layer. Normalization is the transformation of data into a normal distribution with a mean of zero and a standard deviation of one. Since the batch normalization layer functions as a regularizer, it can also be used to meet the requirements of other regularisation techniques.
  6.  <b>Different architecture</b><br>
The problem of vanishing gradient can also be relieved by using an architecture that appears more resistant to this deficiency, like the <em>Long Short Term Memory</em? (LSTM).The LSTM is an RNN architecture that allows the model to maintain the long-distance dependencies and prevent the VG by letting the gradient also flow unchanged. An LSTM unit consists of a cell, an input gate, an output gate, and a forget gate. The unit cell can retain the data for the desired number of time intervals.


  8.  <b>Residual networks</b><br>
  A residual neural network (ResNet) utilize skip connections or shortcuts to skip over some layers. The ResNets are typically implemented with double or three-layer skips, which contains nonlinearities (ReLU) and batch normalization in between. These skip connections helps the model training process to avoid the VG problem.
  
## Exploding Gradient problem
<p>
Figure-7 depicts the exploding gradient in an intuitive manner. Everything appears to work properly until point <em>m</em>, when the gradient begins to decrease from point <em>s</em>. The function's derivative, however, is so steep at point m that the derivative has a magnitude negative value. The gradient is expected to shift right, approaching the global minimum. However, due to the steepness of the derivative, it jumps to point <em>p</em>, causing the global minimum to be missed. This problem results in an unstable network, which is not able to learn from the training data, and at worst results in bad solution contains NaN weight values that can no longer be updated.
</p>

![gradient_descent_0113.png](/mytechblog/images/2022-03-11-DL_gradient_descent/gradient_descent_0113.png "Figure-7, exploding gradient.")


<p>
There are some signals to detect exploding gradients throughout the model training as outlined below:<br>
1.  Your model is struggling to get traction on your training data (poor loss).<br>
2.  A large change in loss occurs over the epochs as a result of the instability in the model.<br>
3.  During training, the model loss becomes NaN.<br>
</p>

<p>
The reasons for exploding gradients:<br>
1.  Incorrect selection of the learning rate, resulting in large weight updates.<br>
2.  Poor data preparation led to large differences in the target variable.<br>
3.  The loss function was poorly chosen, allowing large error values to be calculated.<br>
</p>

### Solutions:
<p>
  Changing the error derivative before propagating it backwards and using it to update the weights is a common solution to exploding gradients. As the error derivative is rescaled, the weights will also be updated, drastically reducing overflows and underflows. Two main methods for updating the error derivative are as follows:<br>
  *   Gradient scaling
  *   Gradient clipping
</p>
