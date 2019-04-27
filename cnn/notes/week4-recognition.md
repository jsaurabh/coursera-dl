## Face Recognition

Face Recognition v Face Verification

### Verification
Input: image, name/ID
Output: determine whether input image is that of the given person

It is a 1:1 problem

### Recognition
Recognition is a much harder problem
Input: image
Output: ID if input is any of the K persons in the database

Both Recognition and Verification are Supervised Learning tasks

## One Shot Learning
Given just one input image, give the correct output reliably. Think of it as achieving 100% accuracy on the test set, wherein the train set consists of a single image. This might not sound hard, but in reality, the input may not be similar to the images in the train set ie. recognize all that is that makes a 'face' of a single person, without having any chance to look at all possible variations of said face.

### Solution? Similarity Function
d(img1, img2) = degree of difference between input images
 
If the inputs are of the same person, the output should be a small number and vice versa.

If d(img1, img2) <= &tau;, then the images are of the same person. Here, &tau; is a hyperparameter that needs to be set during learning.

The above similarity function helps us address the face verification problem

To use this in a face recognition system, the input image is compared with the K images in the database. If any returned value is <= &tau;, this tells us that the person is in the database and has been recognized successfully.

## Possible problems: 
1. Pairwise comparison, not feasible for large datasets?

## Siamese Function
A good method to implement the similarity function is a Siamese network.

A Siamese network consists of a convolutional architecture, sans the final softmax/output layer. The final layer in such an architecture is usually a fully connected layer

Let's consider a 128 feature encoding from the final fully connected layer. Compute such encodings for all images in the database. For a given input image, compute the corresponding feature vectors for the same and use a distance measure between the two to determine if both the images are of the same person or not. 

Say the feature vector for an image in the database is f(x<sup>(1)</sup>) and the feature image for the image at test time is f(x<sup>(2)</sup>)

Use a distance metric d(x<sup>(1)</sup>, x<sup>(2)</sup>) is defined as the norm between the differences of these two feature vectors(also referred to as encodings)

This idea of running CNNs on two different inputs and then comparing their outputs is referred to as a Siamese n/w architecture.

### How to train Siamese Networks?
Parameters of the network define the encoding f(x<sup>(1)</sup>). So, learn the parameters such that for two images of the same person, the distance norm is small and large if not. Simple backpropagation to learn different parameters of the network that minimizes loss can be used to effectively train Siamese architectures.

Speaking of, how is loss measured in such architectures?

## Triplet Loss

Arises from the fact that you'll be looking at 3 images(anchor, positive and negative) at a time. 

We want the distance norm between the anchor and positive image to be small, and smaller than the distance norm between the anchor and negative image.

|| f(A) - f(P)||^2 2 - || f(A) - f(PN||^2 +  &alpha; <= 0 where &alpha; is a hyperparameter. Here, &alpha; is called the margin.

The function of the margin parameter is to push the anchor positive pair and the anchor negative pair away from each other. Triplet loss can be defined as follows:
    
    L(A, P, N) = max(|| f(A) - f(P)||^2 2 - || f(A) - f(PN||^2 +  &alpha;, 0)
and so long as the first term is <=0, the loss is zero and positive otherwise. 

For a dataset with 10k pictures of 1k persons, generate triplets and then train using SGD defined on a cost function that takes as inputs triples of images.

So how does this solve the One-Shot Learning problem? Because of the nature of the loss itself, we'd need to have multiple pictures of the same person(anchor and positives). This can then be extended to one-shot learning where you'd have just a single image as an input.

### How to choose triplets?
Using random choice, d(A, P) + &alpha; <= d(A, N) is easily satisfied. Chances are A and N are very different and as such, the distance will be large and the network won't be able to learn much because the above constraint won't be easily satisfied.

The idea is to choose triplets that are hard to train on, so that the constraint is satisfied. A very promising scenario is when d(A, P) is very close to d(A, N). In such cases, the network will be forced to learn and push 'both images away from each other' such that margin &alpha; is maximized. 

## Face Verification and Binary Classification

Face Recognition as a Binary Classification Problem
Generate embeddings(feature vectors) for input images, and pass it to a logistic regression unit to make a prediction 

The input to the logistic regression unit can be the element wise difference between input images over number of features. This input can have additional parameters such as weights and biases, allowing these parameters to be learned over gradient descent.

Another possible method to calculate the difference can be the &chi;<sup>2</sup> similarity method. Additional details are presented in the paper.