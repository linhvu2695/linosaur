---
date: 2025-01-08 12:26:40
layout: post
title: Transformers - one model to rule them all
subtitle: Transformers are the future. Well, they’re the present at least.
description: We used to have a zoo of many architectures for machine learning models. Now, Transformer is the present. And it's worth knowing pretty much every detail of this model.
image: https://www.chatgptguide.ai/wp-content/uploads/2024/08/image-18.jpg
optimized_image: https://www.chatgptguide.ai/wp-content/uploads/2024/08/image-18.jpg
category: Model
tags:
  - model
  - nlp
  - ai
author: linhvu2695
paginate: true
---

In 2017, eight scientists from Google published a new deep learning architecture called Transformer in the legendary paper <a href="https://arxiv.org/abs/1706.03762">Attention is all you need</a>, based on the attention mechanism proposed in 2014 by <a href="https://arxiv.org/abs/1409.0473">Bahdanau et al</a>  and <a href="https://arxiv.org/abs/1508.04025">Luong et al</a>. The paper's title is a reference to the song "All You Need Is Love" by the Beatles. It is considered a foundational paper in modern artificial intelligence, as the transformer approach has become the main architecture of large language models like those based on GPT. Today, we will explore the intricacies of this groundbreaking architecture, unboxing the magic behind this powerful black box.

To understand this marvel of architecture, let's first take a look at the bird-eye view of this model, starts from this well-known image:
<img src="https://quantdare.com/wp-content/uploads/2021/11/transformer_arch.png">
*Transformer overall architecture*

## Encoder - Decoder
There are 2 main components in the transformer architecture: the **encoder** (on the left) and the **decoder** (on the right). This has always been the fundamental of many language models since the time of RNNs. Input sequences are processed by the encoder, sequentially word by word, turning into vectors (aka **hidden states**) that represent the contextual meaning of the input. These hidden states are then passed through the decoder, which generates the output sequence.
<img src="https://jalammar.github.io/images/t/The_transformer_encoders_decoders.png">
<div style="display: flex; justify-content: center; align-items: center;">
  <video width="800" height="auto" loop autoplay controls>
    <source src="https://jalammar.github.io/images/seq2seq_6.mp4" type="video/mp4">
  </video>
</div>
The heaviest drawback of RNN is the vanishing gradient problem. As the sentence grows longer, the model starts to forget about content from the beginning. This causes the model to struggle to learn long-term dependencies in the input sequences. To address this, **attention** mechanism arrived to save the day.

## Attention
The main idea of attention is to selectively focus on relevant parts of the input sequence when generating the output. To achieve this, all encoder hidden states will be brought over to the decoder, instead of the latest one. The decoder will then be able to attend to the whole input sequence, and focus on the related parts.
<div style="display: flex; justify-content: center; align-items: center;">
  <video width="800" height="auto" loop autoplay controls>
    <source src="https://jalammar.github.io/images/seq2seq_9.mp4" type="video/mp4">
  </video>
</div>

Below picture is also from the paper, showing the attention mechanism for a translation task. The most highlighted part is the term *"the European Economic Area"*, which has been gracefully translated into *"la zone Économique Européenne"*. See how the most relevant term got the most heavy weight on the attention array for a word.
<img src="https://jalammar.github.io/images/attention_sentence.png">

### 1. Self-attention
What we see above is the output sequence attending to the input sequence, or cross-attention (we will talk more on this later). However, attention can even be applied on a sequence itself. This is called self-attention. It allows a sequence to enrich itself, with each token being able to attend to all other tokens in the same sequence, therefore adding a context meaning to its own embedding.
<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1f5eb985-c93d-49a0-8ea4-f48b85e71349_3388x1601.png">

Here is how it works: each token's embedding vector is transformed into three vectors: *Query* (Q), *Key* (K), and *Value* (V). These vectors are derived by multiplying the input embedding matrix with learned weight matrices for Q, K, and V.
* **Query**: represent the token that we want to enrich its meaning.
* **Key**: represent other tokens in the sequence. See how Q & K multiplied together, creating a square matrix of attention weights.
* **Value**: represent the token itself again. This value will eventually attends to other tokens through the above attention weights.
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTDfLojmiBhIBJNk2oymcmiOugCAV6YfVnwXtUKCJhdshVm-HR7cVvdF2NnduDKWjtEgw&usqp=CAU">
*The division by dimensionality is to prevent large variance in dot-product values, which can negatively impact the stability of the softmax function.*

The whole above process is also known as *Scaled Dot-Product Attention*.
<img src="https://jalammar.github.io/images/t/transformer_self-attention_visualization.png">
*See how the word "it" focuses on "The animal", implying that these two entities are referring to the same thing*

### 2. Multi-head Attention
Why limit ourselves to just one set of attention Q, K, V? Multi-head attention allows the sequence to enrich itself - in multiple ways! Each token can learn about its context on different syntactic and semantic areas: word ordering, adjective & adverb roles, entity relations, etc.
<img src="https://data-science-blog.com/wp-content/uploads/2022/01/mha_img_original.png">

<img src="https://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png">
*On another attention head, the word "it" now focus on it's state "tired"*

### 3. Masked attention
Imagine you writing a sentence. Each word you are writing out can only be understood in the context of what you've written before - your sentence is stopping at where you are writing. That's the concept of masked attention. When calculating self-attention, we will mask out the future tokens, to prevent information leakage. And we only apply masking for decoder input, as it represent the process of "writing" out. For encoder input, it represent the process of "reading" in, which is totally fine if we put a word in the context of a whole sentence.
<img src="https://res.cloudinary.com/dptj6j9y9/image/upload/v1739030594/e6f9204c1ebd4fddba9c37c8b66faf04_xcytza.png">

### 4. Cross-attention
The function of cross-attention is similar to that of self-attention, only the Q, K, V inputs are plugged in differently. Decoder input (Q) is combined with Encoder's hidden states (K) to produce the attention weights. After that, Encoder's hidden states (V) will then utilize that matrix to produce output.
<img src="https://media.licdn.com/dms/image/v2/D5622AQES_5kWgZ8O4Q/feedshare-shrink_800/feedshare-shrink_800/0/1719243502822?e=2147483647&v=beta&t=RKq2R7bzBIufBucIvrfLTSi6nvq2N4vIrpn5cGaN6-s">

The below image illustrates clearly how self-attention and cross-attention work together in the Transformer architecture:
<img src="https://res.cloudinary.com/dptj6j9y9/image/upload/v1739031859/0_R_rhiu44twW2iz0W_aep6ui.png">

## Positional Encoding
One thing that’s missing from the model as we have described it so far is a way to account for the order of the words in the input sequence.
To address this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.
<img src="https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png">

## Residuals
One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a layer-normalization step.
<img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png">

This goes for the sub-layers of the decoder as well. If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:
<img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png">

## Finals
That's it! We've covered the core concepts of the Transformer architecture, including the encoder, decoder, attention mechanisms, positional encoding and residuals. Final vectors from Decoder will just go through an MLP and a softmax layer to output the final token.

The introduction of Transformer is truly a historical mark for the AI world. It's a groundbreaking architecture that has significantly improved the performance of various NLP tasks, such as machine translation, text summarization, and question answering. As of 2024, the paper has been cited more than 140,000 times.

Some notable resources to learn more about Transformers:
* Tensor2Tensor <a href="https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb">Jupyter Notebook</a>
* Interactive model <a href="https://poloclub.github.io/transformer-explainer/">Transformer Explainer</a>











