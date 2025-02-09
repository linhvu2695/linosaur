---
date: 2025-01-09 11:16:40
layout: post
title: A Brief History of CNNs
subtitle: Convolutional Neural Networks - grant computers an eye to see the world.
description: Computer Vision has made fascinating journey, from the early foundations in 1980s to the present powerful models that can perceive almost anything in the real world.
image: https://res.cloudinary.com/dptj6j9y9/image/upload/v1738773481/Blog-Banner-Computer-Vision_mq7nlh.jpg
optimized_image: https://res.cloudinary.com/dptj6j9y9/image/upload/v1738773481/Blog-Banner-Computer-Vision_mq7nlh.jpg
category: Model
tags:
  - model
  - computer vision
author: linhvu2695
paginate: true
---

Understanding visual data is a hugely challenging task for computer programs. Before CNNs, the standard way to train a neural network to classify images was to flatten it into a list of pixels and pass it through a feed-forward neural network to output the image’s class. The problem with flattening the image is that the essential spatial information in the image is discarded. In 1989, Yann LeCun and team introduced Convolutional Neural Networks — the backbone of Computer Vision research for the last 15 years! Unlike feedforward networks, CNNs preserve the 2D nature of images and are capable of processing information spatially!

Today, we are going to go through the history of CNNs — starting from those early research years in the 90’s to the golden era of the mid-2010s when many of the most genius Deep Learning architectures ever were conceived, and finally discuss the rising trend of Transformers in which Computer Vision is catching on.
<img src="https://res.cloudinary.com/dptj6j9y9/image/upload/v1738774030/1_R9YqQdYqHZsTw9P7Uw5AOw_djkyse.webp">
*The history of CNN models from 1989 to today*

# Basic Concetps
## Kernel
At the heart of a CNN is the convolution operation, based on a matrix called kernel. We scan the kernel across the image and calculate the dot product of the kernel with the image at each overlapping location.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/14ZBZhhftMDbMYnsdxzzPfg-1.gif">
*How Convolution works – The kernel slides over the input image and calculates the overlap (dot-product) at each location – outputting a feature map in the end!*

In a convolution layer, we train multiple filters that extract different feature maps from the input image. When we stack multiple convolutional layers in sequence with some non-linearity, we get a convolutional neural network (CNN).
So each convolution layer simultaneously does 2 things —
1. spatial filtering: using the kernel to extract spatial features. A new pixel now represent information from a localized region
2. channel mixing: output a new set of channels. A new channel now represents the image’s property (eg., corners, boundaries…), not just a simple RGB value.
90% of the research in CNNs has been to modify or to improve just these two things.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1Xu0kxaS40fyljAKMvp4t_Q-1536x864.jpeg">
*The two main things CNN do*

# The 1989 Paper
<a href="http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf">This 1989 paper</a> taught us how to train non-linear CNNs from scratch using backpropagation. They input 16×16 grayscale images of handwritten digits (the infamous MNIST), and pass through two convolutional layers with 12 filters of size 5×5. The kernels also move with a stride of 2 during scanning. **Strided-convolution is useful for downsampling the input image**. After the conv layers, the output maps are flattened and passed through two fully connected networks to output the probabilities for the 10 digits. Using the softmax cross-entropy loss, the network is optimized to predict the correct labels for the handwritten digits. After each layer, the tanh nonlinearity is also used — allowing the learned feature maps to be more complex and expressive. With just 9760 parameters, this was a very small network compared to today’s networks which contain hundreds of millions of parameters.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1ct0jgrMwlxJRSBtHHhCI7Q-1-1536x864.jpeg">
*The OG CNN architecture from 1989*

# LeNet-5 (1998) - The Pioneer
In 1998, Yann LeCun and team published the <a href="http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf">LeNet 5</a> — a deeper and larger 7-layer CNN model network, as they applied a backdrop style to Fukushima’s CNN architecture. LeNet5 has around 60,000 parameters, and it can be considered the **standard template** for all modern CNNs as all of them follow the pattern of stacking convolutional and pooling layers.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1AxSYBoE_XlGe_XeUQ-COdw-1-1536x506.png">

**Max Pooling** will help downsample the image by grabbing the maximum values from a 2×2 sliding window. 
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1_gjfM9rU5DfvnaI2OUIozg-1-1536x864.jpeg">
*How max pooling works*

Notice when you train a 3×3 conv layer, each neuron is connected to a 3×3 region in the original image — this is the neuron’s **local receptive field** — the region of the image where this neuron extracts patterns from.
When we pass this feature map through another 3×3 layer , the new feature map indirectly creates a receptive field of a larger 5×5 region from the original image. Additionally, when we downsample the image through max-pooling or strided-convolution, the receptive field also increases — making deeper layers access the input image more and more globally.
For this reason, earlier layers in a CNN can only pick low-level details like specific edges or corners, and the latter layers pick up more spread-out global-level patterns.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1E6PIrM-QbrYcdF_sUfBG7A-1-1536x864.jpeg">

# The Draught (1998-2012)
As impressive Le-Net-5 was, researchers in the early 2000s still deemed neural networks to be computationally very expensive and data hungry to train. There was also problems with **overfitting** — where a complex neural network will just memorize the entire training dataset and fail to generalize on new unseen datasets. The researchers instead focused on traditional machine learning algorithms like support vector machines (**SVMs**) that were showing much better performance on the smaller datasets of the time with much less computational demands.

The <a href="https://www.image-net.org/index.php">ImageNet dataset</a> was open-sourced in 2009 — it contained 3.2 million annotated images at the time covering over 1000 different classes. Today it has over 14 million images and over 20,000 annotated different classes. Every year from 2010 to 2017 we got this massive competition called the ILSVRC where different research groups will publish models to beat the benchmarks on a subset of the ImageNet dataset. In 2010 and 2011, traditional ML methods like SVMs were winning — but starting from 2012 it was all about CNNs. The metric used to rank different networks was generally the **top-5 error rate** — measuring the percentage of times that the true class label was not in the top 5 classes predicted by the network.

# AlexNet (2012) - The Breakthrough
<a href="https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf">AlexNet</a>, introduced by Dr. Geoffrey Hinton and his team was the winner of ILSVRC 2012 with a top-5 test set error of 17%. It has over 60 million parameters. On the date of its publication, the authors of AlexNet believed that it was the largest neural network on the subsets of ImageNet. Here are the three main contributions from AlexNet:

## Multi-scaled kernels
AlexNet trained on 224×224 RGB images and used multiple kernel sizes in the network — an 11×11, a 5×5, and a 3×3 kernel. Models like Le-Net 5 only used 5×5 kernels. Larger kernels are more computationally expensive because they train more weights, but also capture more global patterns from the image. Because of these large kernels, AlexNet had over 60 million trainable parameters. All that complexity can however lead to overfitting.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1UCkgETIJwxuT92iqzBCqpQ-1-1536x864.jpeg">

## Dropout
To alleviate overfitting, AlexNet used a regularization technique called Dropout. During training, a fraction of the neurons in each layer is turned to zero. This prevents the network from being too reliant on specific neurons or groups of neurons for generating a prediction and instead encourages all the neurons to learn general meaningful features useful for classification.
<img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Dropout_mechanism.png">

## RELU
AlexNet also replaced tanh nonlinearity with ReLU. RELU is an activation function that turns negative values to zero and keeps positive values as-is. The tanh function tends to saturate for deep networks because the gradients get low when the value of x goes too high or too low making optimization slow. RELU offers a steady gradient signal to train the network about 6 times faster than tanH.
<img src="https://www.researchgate.net/publication/354971308/figure/fig1/AS:1080246367457377@1634562212739/Curves-of-the-Sigmoid-Tanh-and-ReLu-activation-functions.jpg">

# GoogleNet (2014) - The Inception
In 2014, GoogleNet paper got an ImageNet top-5 error rate of 6.67%. The core component of GoogLeNet was the **inception module** – likely named due to a quote from the movie Inception (“We need to go deeper”), which launched a viral meme. Each inception module consists of parallel convolutional layers with different filter sizes (1×1, 3×3, 5×5) and max-pooling layers. Inception applies these kernels to the same input and then concats them, combining both low-level and medium-level features.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1TwHMa1AfX2gAv_Ubz_dIwA-1-1536x864.jpeg">
*An Inception module*

## Pointwise convolution
GoogleNet used a special technique – the 1×1 convolutional layer. Each 1×1 kernel first scales the input channels and then combines them. 1×1 kernels multiply each pixel with a fixed value. While larger kernels like 3×3 and 5×5 kernels do both spatial filtering and channel combination, 1×1 kernels are only good for channel mixing, and it does so very efficiently with a **lower number of weights**. For example, A 3-by-4 grid of 1×1 convolution layers trains only (1×1 x 3×4 =) 12 weights — but if it were 3×3 kernels — we would train (3×3 x 3×4 =) 108 weights.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1iD0MYgFo7jdshDfV8ZDy2g-1-1536x864.jpeg">

# VGGNet (2014)
<a href="https://arxiv.org/abs/1409.1556">The VGG Network</a> claims that we do not need larger kernels like 5×5 or 7×7 networks and **all we need are 3×3 kernels**. 2 layer 3×3 convolutional layer has the same receptive field of the image that a single 5×5 layer does. Three 3×3 layers have the same receptive field that a single 7×7 layer does. But with fewer parameters! Training with deep 3×3 convolution layers became the standard for a long time in CNN architectures.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1feGtykjciBQjN-jBXX0QvQ-1-1536x864.jpeg">

## Batch Normalization
Deep neural networks can suffer from a problem known as **Internal Covariate Shift** during training. Since the earlier layers of the network are constantly training, the latter layers need to continuously adapt to the constantly shifting input distribution it receive from the previous layers.

<a href="https://arxiv.org/pdf/1502.03167">Batch Normalization</a> aims to counteract this problem by normalizing the inputs of each layer to have zero mean and unit standard deviation during training. A batch normalization (BN) layer can be applied after any convolution layer. It makes the network robust to the initial weights of the network.

# ResNet (2016) - The Briliant
Imagine you have a shallow neural network that has great accuracy on a classification task. Turns out that if we added 100 new convolution layers on top of this network, the training accuracy of the model could go down!
This is quite counter-intuitive because all these new layers need to do is copy the output of the shallow network at each layer — and at least be able to match the original accuracy. In reality, deep networks can be notoriously difficult to train because gradients can saturate or become unstable when backpropagating through many layers. With Relu and batch norm, we were able to train 22-layer deep CNNs at this point — the good folks at Microsoft introduced <a href="https://arxiv.org/abs/1512.03385">ResNets</a> in 2015 which allowed us to stably train 150 layered CNNs. What did they do?

## Residual learning
The input passes through one or more CNN layers as usual, but at the end, the original input is added back to the final output. These blocks are called residual blocks because they don’t need to learn the final output feature maps in the traditional sense — but they are just the residual features that must be added to the input to get the final feature maps. *If the weights in the middle layers were to turn themselves to ZERO, then the residual block would just return the identity function — meaning it could easily copy the input X.*
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1XVFU7lfuzC_gkpLcasMolw-1-1536x864.jpeg">

With this simple but brilliant idea, ResNets managed to train a 152-layered model that got a top-5 error rate that shattered all previous records!
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1guSWB-JPErEPi4B17YpSJw-1.png">

# MobileNet (2017) - The Optimizer
As we have learned from above section, convolution layers do two things –1) spatial filtering and 2) combining them channel-wise. The MobileNet paper uses **Depthwise Separable Convolution**, a technique that separates these two operations into two different layers — Depthwise Convolution for filtering and pointwise convolution for channel combination.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1fkqAuw1LpGwF0yDzCiq9uA-1-1536x864.jpeg">

Separating the filtering and combining steps like this **drastically reduces the number of weights**, making it super lightweight while still retaining the performance.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1eVjS5PdjSw1uQp4MpAbsRQ-1-1536x864.jpeg">

# MobileNetV2 (2019)
In 2018, MobileNetV2 improved the MobileNet architecture by introducing two more innovations

## Linear Bottlenecks
MobileNetV2 uses a 1×1 pointwise convolution for dimensionality reduction, also called **linear bottlenecks**. These bottlenecks don’t pass through RELU and are instead kept linear. RELU zeros out all the negative values that came out of the dimensionality reduction step — and this can cause the network to lose valuable information especially if a bulk of this lower dimensional subspace was negative. Linear layers prevent the loss of excessive information during this bottleneck.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1xyvveiqDqMApN6bvU-eD2Q-1-1536x864.jpeg">
*The width of each feature map is intended to show the relative channel dimensions.*

## Inverted Residuals
Generally, residual connections occur between layers with the highest channels, but the authors add shortcuts between the bottlenecks layers. The bottleneck captures the relevant information within a low-dimensional latent space, and the free flow of information and gradient between these layers is the most crucial.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1riEtQs0TtxQPbW0tN_ewXA-1-1536x864.jpeg">

# Vision Transformers (2020) - The Game Changer
Vision Transformers or ViTs established that transformers can indeed beat state-of-the-art CNNs. Catching up with the brewing revolution storm of NLP, applying the **Transformers** and **Attention** mechanisms that provide a highly parallelizable, scalable, and general architecture for modeling sequences.
Patch Embeddings & Self Attention

The input image is first divided into a sequence of fixed-size patches. Each patch is independently embedded into a fixed-size vector either through a CNN or passing through a linear layer. These patch embeddings and their positional encodings are then inputted as a sequence of tokens into a self-attention-based transformer encoder. Self-attention models the relationships between all the patches, and outputs new updated patch embeddings that are contextually aware of the entire image.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/18tJJn-9g49K0pwz5z8g8nQ-1-1536x864.jpeg">
*Each self-attention layer further contextualizes each patch embedding with the global context of the image*

Where CNNs introduce several inductive biases about images, Transformers do the opposite — No localization, no sliding kernels — they rely on generality and raw computing to model the relationships between all the patches of the image. The Self-Attention layers allow global connectivity between all patches of the image irrespective of how far they are spatially. Inductive biases are great on smaller datasets, but the promise of Transformers is on massive training datasets, a general framework is going to eventually beat out the inductive biases offered by CNNs.
<img src="https://towardsdatascience.com/wp-content/uploads/2024/06/1Orozl7-yYB9Gfe1rI8NXGg-1-1536x864.jpeg">

# Final Thoughts
The history of CNNs teaches us so much about Deep Learning, Inductive Bias, and the nature of computation itself. It’ll be interesting to see what wins out in the end — the inductive biases of ConvNets or the Generality of Transformers. Whatever happens, it is a blessings to be able to witness all the amazing evolution going on in the area of Computer Vision.


