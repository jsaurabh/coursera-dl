# Neural Stlye Transfer

## What are deep Convnets learning?
Or more intuitively, what are the shallow and deeper layers visualizaing

For a unit in each layer, find the image patches that maximize that hidden unit's activation. Repeat for all other units and plot the resulting activations

In the paper titled Visualizing and Understandinf Convolutional Neural Nets, the authors plotted maximal activations for 9 hidden units for each layer. You can definitely plot activations for more hidden layers. The activations, when plotted, helped in understanding what features each layer of a convnet learns, going from simple edges and lines in earlier layers to more complex textures and even text in the later layers.

## Cost Function
For Neural Style Transfer, the input image is said to be the Content C, the style is called Style S and the output is the generated image G.

What is the cost function for such an application?
There are two parts to minimizing the cost function using gradient descent, one is the cost function for the content C and the second one is for the style S

J(G) = &alpha; J(C, G) + &beta; J(S, G) where &alpha; and &beta; are the two hyperparameters that specifying the relative weighin cost between the content and style. 

How to find the generated image G

1. Initialize G randomly with the target dimension
2. Minimize J(G) by gradient descent
    G -= partial derivative of J(G)

As you minimize, the pixel values of G change and over multiple iterations, they converge to the required styled image.

## Content Cost Function
Choose a hidden layer l

l is chosen to be somewhere in the middle of the network, neither too shallow nor too deep. If l is chosen to be one of the shallower layers, the generated image will end up having pixel values similar to the content image, whereas if l is chosen to be too deep, the generated image will have a non-meaningful replication of the content image. ##WHY? 

Use a ConvNet, given a content and generated image
Let a<sup>l(c)</sup> and a<sup>l(g)</sup> be the activations of the content and generated iamge at layer l. If these two activations are similar, then the two images have similar content.

J(C,G) = ||a<sup>l(c)</sup> - a<sup>l(g)</sup>||<sup>2</sup>

Unroll into vectors, and take the squared of the L2 norm(np.linalg.norm). It is really the element wise  sum of squared of differences between the activations of the content and generated image at layer l.

On performing gradient descent to minimize J(G), it will incentivize the image to find G so that the hidden layer activations for the same are similar to the ones for the content image.

## Style Cost Function
What is Style?

Choose a hidden layer l. Style is defined as the correlation between different channels for a hidden layer l

Now, how does this work? By calculating the correlation across channels, you get a sense of what textures(features) occur together, how often and how often do they not occur together. Using gradient descent to minimize this cost leads to a generated image that maximally captures this correlation and hence is in the style of the style image S.

Formally speaking, 

Let a<sup>[l]</sup><sub>i,j,k</sub> be the activation across i, j and k at layer l. Here i is the height, j is the width and k is the number of channels.

To measure the correlation, let G<sup>[l]</sup> = n<sub>c</sub><sup>[l]</sup> * n<sub>c</sub><sup>[l]</sup>. This makes sense because to measure correlation, you'd need a (nc,nc) square matrix to get the correlation across all possible combinations.

G<sup>[l]</sup><sub>k, k'</sub> will measure the correlation between activations in k and k' as formulated above. 

G<sup>[l]</sup><sub>k, k'</sub> is then defined as summing over i and j the product between activations in k and k'

This is actually not correlation but the unnormalized cross-covariance between the elements in k and k'. If the product is higher,
intuitively speaking, the two neurons are 'correlated'.

Repeat the process above for the style image S and the generated image G. 

Using G(S) and G(G), the style cost function J(S,G) is the sum of squares of the element-wise differences 

The overall style cost function is the sum of style cost over different layers, leading to more visually pleasing results by allowing for lower and higher level 'correlations' to be accounted in the style.