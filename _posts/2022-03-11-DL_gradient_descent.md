# Gradient Descent


Here's the table of contents:

1. TOC
{:toc}

## Overview
<em>Gradient descent</em> is by far the most crucial mathematical concept used in deep learning algorithms, and it is fair to say that deep learning would not be possible without it. The gradient vector can be interpreted as the "direction and rate of fastest increase". If the gradient of a function is non-zero at a point <em>p</em>, the direction of the gradient is the direction in which function increases most quickly from <em>p</em>, and the magnitude of the gradient is the rate of increase in that direction, the greatest absolute directional derivate. Furthermore, the gradient is the zero vector at a point if and only if it is a stationary point, where the <em>derivative</em> vanishes (<a href="https://en.wikipedia.org/wiki/Gradient">source</a>). From the calculus we know that a local minimum of a function happens in the point in which the function's derivative is equal to zero (as the first order necessary condition). This is the reason that gradient plays a fundamental role in optimization theory, where it is used to minimize or maximize a function by gradient descent or gradient ascent respectively. As a result, we now understand why gradient descent is essential to minimizing the loss of a deep learning model.

The gradient of a function is closely related to its derivative. 
Basically, the gradient of a function <em>f</em> is the dual to its <em>total derivative</em>, or in other words, they are related in that the <em>dot product</em> of the gradient of <em>f</em> at the point <em>p</em> with another tangent vector <em>v</em> equals the <em>directional derivative</em> of <em>f</em>, which is depicted as below:
![gradient_derivative_01.svg](/mytechblog/images/2022-03-11-DL_gradient_descent.md/gradient_derivative_01.svg)


