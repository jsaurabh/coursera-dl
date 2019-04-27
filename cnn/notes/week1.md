## Padding 
Modification to images

For an input 6x6 image, convolved with a 3x3 filter, the output is a 4x4
In general, for image of shape nxn, with filter shape of fxf, the shape of the output image is of shape (n-f)+1

## Problems
1. Basically, for every convolutional operation the image shrinks. 
2. The edge pixels influence only one region in the convolution output, whereas the centered ones have a say in more output regions(throwing away information from edges)

## FIX? 
Pad the image with a border of say 1px. Essentially, the image then becomes 8x8, and convolving with 3x3, we get 6x6 output image,which is the same dimension as the input image. Essentially, this means all the edge pixels influence the output image pixels. This fixes both problems mentioned above.

## Convention is to pad with zeros, so as not to add any noise to the image

For padding border p, output is of shape (n+2p-f+1, n+2p-f+1) 

## How Much to Pad?

Valid Convolution - p =0. Output of shape (n-f+1, n-f+1)
Same Convolution - Pad such that output size is same as input size. Determine value of p from experssion p = (f-1)/2 

### Side Note:By convention, filter size(f) is usually odd. Why?  

For even f, you'd need asymmetric padding.
For odd dimension filters, there's a distinct central pixel which is not the case for even numbered filters

## Strided Convolutions

Stride = step size for each successive convolution operation(horizontally and vertically)

Shape of output image
floor((n-f+2p)/s) +1 where s is stride 
For n=7,7 f = 3,3 p=0 and s = 2, the output shape is 3,3

There will be cases where (n-f+2p)/s is not an integer. In such cases, round down to nearest integer. This is done because a fractional corresponds to a part of the image that doesn't fit within the image+padding and filter dimensions. The output must lie entirely within the above dimensions.

Convolution is done without flipping the kernel?

## Convolution over Volume

3D Volume(eg. RGB image)

Input - 6x6x3 (h, w, n) where n is number of channels
Filter - 3x3x3 (h, w, m) where m is number of channels
Critical that m==n
    
Output - 4x4 (2D output image of shape (n-f+1, n-f+1))

For all 3 channels in the filter, convolve it with the input image over all three channels, add it and that becomes each individual pixel in the output image. Total multiplications will be (h*w*n)

For using individual filters(say just detect edges in red channel), you could set the filter values for B and G channels in the filter to zero. Choose appropriate values for each filter based on the feature to detect from the input image

Multiple Filters - for detecting multiple features simultaneously
Multiple filters will lead to different outputs, all of the same shape. To get the final output, stack the individual filters outputs together 

In general, for nxn output dimension of individual output with k simultaneous filters, the shape of the cumulative output is nxnxk (3D output image)
Very powerful idea, allowing detection of multiple features with output having number of channels == number of features detected.
Channels represents the same idea as depth of the filter.

## A Single Layer of a Convolutional Neural Network

Input image - 6x6x3
Filter1 - 3x3
Filter2 - 3x3

temp1 - 4x4
temp2 - 4x4
Output1 - ReLU(temp1 + bias1)
Output2 - ReLU(temp2 + bias2)

Output n - ReLU(temp n + bias n)

Output - stack Output[1..n]

## Types of Layers

Pooling Layer(POOL) Reduce size of representation and make feature detection more robust
    
What is Pooling?

Choose a stride and filter size, and for each corresponding output pixel, the element chosen will be the (max, avg) of the constituent filter stride elements

Why Max Pooling

The assumption is that a high value in the input pixels corresponds to the presence of a feature after activations in some layer of the network. Max pooling will thus always select that feature weights to be passed on to the next network

**There is nothing for gradient descent to learn in Max Pooling, after filter size and stride size are chosen ie no filter weights as output is just derived from the input, without any linear combination

Max Pool on 3D input(independent computation on each channel)

Perform max pool computation on each channel independently, and then stack the respective channels to preserve output shape
Max Pool generally doesn't take any padding(p=0)
    
Average Pooling
Take average of each filter size, over all strides.

### Unless you are really deep in the network and want to collapse the input size, max pooling is preferred.


## Fully Connected(FC) 
Every element is connected to every other element in the next layer, giving a fully linear combination of the current and previous layer

## Why Convolutions

Parameter sharing
    A feature detector useful in one part is probably useful in another part of the image.

Sparsity of connections
    Each output unit is dependent on only a small number of input pixels

Invariant to translations