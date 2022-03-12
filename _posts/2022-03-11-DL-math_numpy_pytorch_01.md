# Math, NumPy, and PyTorch for Deep Learning


Here's the table of contents:

1. TOC
{:toc}

## Linear Algebra Terminology
Figure-1 shows some linear algebra terminilogies that are very commonly used in deep learning.

![linear_algebra_terminology_01.jpg](/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/linear_algebra_terminologies_01.png 
  "Figure-1, Frequently used linear agebra terminologies in deep learning.")
  
{% include definition.html text="
<b>Scalar</b>: In vector space, a <em>scalar</em> is used to define elements of a field. In linear algebra, real numbers or generally, elements of a field are called scalars and relate to vectors in an associated vector space.<br>
<b>Vector</b>:  A quantity described by multiple scalars, such as having both direction and magnitude, is called a <em>vector</em>.<br>
<b>Matrix</b>: In mathematics, a <em>matrix</em> (plural matrices) is a rectangular array or table of numbers, symbols, or expressions, arranged in rows and columns, which is used to represent a mathematical object or a property of such an object.<br>
<b>tensor</b>: In mathematics, a <em>tensor</em> is an algebraic object that describes a multilinear relationship between sets of algebraic objects related to a vector space.<br>
<small><em>source: Wikipedia.</em></small>
" %}

Figure-2 shows a Euclidean vector, which its coordinates x and y are scalars, but v is a vector that defined by scalars x and y.<br>
![Vector_components.svg.png](/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/Vector_components.svg.png 
  "Figure-2, Euclidian vector defined by scalars x and y.")
  
## Storing Images on Computers
For the application of DL in computer vision, the main problem is how computers interpret images. In order to accomplish this, an image must be represented by digits that are stored in an easy-to-manage data structure for computers. The key to solving this problem is using matrices and tensors. A grayscale image can be stored in a matrix of digits, each element of the matrix representing a pixel. A pixel has an integer value between 0 and 255, i.e., 0 corresponds to black, and 255 corresponds to white. The color value in grayscale images represents its brightness. The more data points in the matrix, the more detailed the image will be (higher resolution image). Figure-3, <a href="https://medium.com/analytics-vidhya/computer-vision-what-how-why-380607f0bd64">source</a>, illustrates intuitively transforming an image from what humans can see, i.e., leftmost image, to what computers can understand for computers, i.e., rightmost image. This image in the center combines two concepts, which might represent one of the capabilities of a DL model.<br>

![image_transforming_01.png](/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/image_transforming_01.png 
  "Figure-3, image transformation.")

Figure-4, <a href="https://lisaong.github.io/mldds-courseware/01_GettingStarted/numpy-tensor-slicing.slides.html">source</a>, shows that how a colored image is strored in a tensor, which is here a 3D tensor and basically functions like a stacked matrices, each of which contains a color channel for red, blue and green. To understand how complicative image processing and deep learning in computer vision is, you should consider that processing a 4K image entails enormous computations over 25 million pixels (3840 x 2160).
![colored_image_transforming_01.png](/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/colored_image_transforming_01.png 
  "Figure-4, colored image transformation.")
  

<table>
  <tr>
    <th>Object</th>
    <th>
    </th>
    <th>![vector.png](/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/vector.png)
    </th>
    <th></th>
  </tr>
  <tr>
    <td>Math</td>
    <td>Scalar</td>
    <td>Vector</td>
    <td>Matrix</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>NumPy</td>
    <td>Array</td>
    <td>Array</td>
    <td>ND array</td>
    <td>ND array</td>
  </tr>
    <tr>
    <td>PyTorch</td>
    <td>Tensor</td>
    <td>Tensor</td>
    <td>Tensor</td>
    <td>Tensor</td>
  </tr>
</table>
