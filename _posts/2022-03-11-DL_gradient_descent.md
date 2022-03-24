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
This section shows, through an example, how to minimize a function using tracking its gradient and find a near minimum solution and the value of variable at the function's minimum. The funciton is as below:
<p><center>y = f(x) = 3x<sup>2</sup> + 3x + 4</center></p>
