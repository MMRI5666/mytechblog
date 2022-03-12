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

<br>
## Data Types
<table>
  <tr style="border-color: bcbca9;">
    <th style="background-color:#ddddd3;">Object</th>
    <th style="background-color:#ddddd3;"><b>7</b></th>
    <th style="background-color:#ddddd3;"><img src="/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/vector.png" alt="vector.png"></th>
    <th style="background-color:#ddddd3;"><img src="/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/matrix.png" alt="matrix.png"></th>
    <th style="background-color:#ddddd3;"><img src="/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/tensor.png" alt="tensor.png"></th>
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


## Converting Reality to Numbers
There are two type main types of reality:
<ul>
  <li><b>Continuous</b><p>Continuous reality is in form of numeric and can 
  contain many (or possibly infinite) distinct values, e.g., height, exam 
  scores, income, review score</p>
  </li>
  <li><b>Categorical</b><p>Reality of this type represents discrete values 
    and can contain limited and specific distinct values, such as pet types 
    (cat or dog), and disease diagnosis (true or false).</p>
    <p>There are two approaches for representing categorical data as:
    <ul>
      <li><b>Dummy-coding</b>
        <p>
          This approach, which is commonly used by most classical machine learning methods, converts a vector of categorical values to a binary vector and assigns the existing values a label of 0 or 1 (true or false), e.g., exam result (pass/fail), house (sold or in-market), fraud detection (1 and 0 or normal and fraudulent transaction).
        </p>
      </li>
      <li><b>One-hot encoding</b>
        <p>
           The functionality of this approach is basically similar to <em>Dummy-coding</em>, but applies to multi-categorical values and creates a matrix, rather than a vector, to wrap up the assigned labels. In this matrix structure, columns and rows correspond to the categories and observations respectively. 
        </p>
      </li>

    </ul>
    </p>
  </li>
</ul>

## Tranposing Vectors and Matrices
In linear algebra and deep learning, transposing is one of the most commonly used operators. In vectors or matrices, this operator changes the orientation of the rows to columns and vice versa without affecting the values.
    
![transpose_01.png](/mytechblog/images/2022-03-11-DL-math_numpy_pytorch_01/transpose_01.png 
  "Figure-5, transposing a matrix.")
  
  {% include note.html text="The transpose of a transposed vector/matrix is exactly the same as the original vector/matrix" %}

The following codes show how to use NumPy and PyTorch to transpose:
*   In NumPy:

```python
  # import libraries<br>
  import numpy as np<br>
  import torch<br>
  
  ls = [ [1, 2, 3, 4], [5, 6, 7, 8]]
  ls
```
    [[1, 2, 3, 4], [5, 6, 7, 8]]
  
```python
# create a matrix
mx = np.array(ls)
# transpose the matrix
mxT = mx.T
print(mxT)

# transpose the transpose
mxTT = mxT.T
print(mxTT)
```
    [[1 5]
    [2 6]
    [3 7]
    [4 8]]
    [[1 2 3 4]
    [5 6 7 8]]

*   In PyTorch:

```python
# create a tensor
tns = torch.tensor(ls)
print(tns)

# transpose the tensor
tnsT = tns.T
print(tnsT)

# transpose the transpose
tnsTT = tnsT.T
print(tnsTT)
```
    tensor([[1, 2, 3, 4],
          [5, 6, 7, 8]])
    tensor([[1, 5],
          [2, 6],
          [3, 7],
          [4, 8]])
    tensor([[1, 2, 3, 4],
          [5, 6, 7, 8]])
          
  {% include note.html text="In NumPy and PyTorch there are 
  functions to calculating transpoe, nemaly <em>np.transpose()</em> 
  and <em>torch.transpose()</em>, but the forms 
  <em>narray.T</em> and <em>tensor.T</em> are more commonly used" %}

<pre class="output_area">
  sdsdfsfsdfsdfsdfsdfsdfgasdfgsdfgsdfgsdfgdasfg
</pre>
