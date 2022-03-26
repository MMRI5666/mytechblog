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

<br><br><br>

![gradient_derivative_01.svg](/mytechblog/images/2022-03-11-DL_gradient_descent/2022-03-26_11-50-39_1.png)
