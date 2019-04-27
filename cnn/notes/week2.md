## Studying Classical Convolutional Neural Network Architectures

Let's start off with the classic LeNet-5 architecture by Yann LeCun, back in 1985

The input to the network is a 32x32x1 image of a handwritten digit, with the aim of classifying it to its correct label. The peculiar shape is because the network is trained on grayscale image, leading to 1 channel

Input : (32, 32, 1) grayscale image 
L1 : f1(5, 5) with stride 1 and 6 channels
Average Pool(L1)

A1 : (28, 28, 6) grayscale image
L2 : f2(2, 2) with stride 2 and 6 channels

A2 : (14, 14, 6) grayscale image
L2 : f3(5, 5) with stride 1 and 16 channels
Average Pool(L2)

A3 : (10, 10, 16) grayscale image
L3 : f4(2, 2) with stride 2 and 16 channels

A4 : Fully Connected Layer with 120 units

A5 : Fully Connected Layer with 84 units

Output : variable y^ which could take one of 10 possible values for digits from 0-9

The network used the sigmoid activation function, and made use of average pooling rather than maxpool which is the norm today. Another interesting tidbit is that the network recommended non-linearities after avgpool.

For reference and implementation details, check out the paper linked below

// Le-Net5 Paper link in Markdown

Now, moving on more than 2 decades, let's review AlexNet, named after Alex Krizhevsky who was the lead author of the paper