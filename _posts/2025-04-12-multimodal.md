---
date: 2025-04-12 13:45:12
layout: post
title: "Multimodal - perceiving the world"
subtitle: Humans can read, talk, and see. We listen to music to relax and watch out for strange noises to detect danger. Being able to work with multimodal data is essential for us or any AI to operate in the real world.
description: Humans can read, talk, and see. We listen to music to relax and watch out for strange noises to detect danger. Being able to work with multimodal data is essential for us or any AI to operate in the real world.
image: https://i.insider.com/67d9cdaf69253ccddf999d59?width=700
optimized_image: https://i.insider.com/67d9cdaf69253ccddf999d59?width=700
category: model
tags:
  - model
  - ai
author: linhvu2695
paginate: true
---
For a long time, each ML model operated in one data mode (aka *modality*) – text (translation, language modeling), image (object detection, image classification), or audio (speech recognition). However, natural intelligence is not limited to just a single modality. Humans can read, talk, and see. We listen to music to relax and watch out for strange noises to detect danger. Being able to work with multimodal data is essential for us or any AI to operate in the real world.

Incorporating additional modalities to LLMs (Large Language Models) creates LMMs (Large Multimodal Models). Not all multimodal systems are LMMs. For example, text-to-image models like Midjourney, Stable Diffusion, and Dall-E are multimodal but don’t have a language model component. Multimodal can mean one or more of the following:

* Input and output are of different modalities (e.g. text-to-image, image-to-text)
* Inputs are multimodal (e.g. a system that can process both text and images)
* Outputs are multimodal (e.g. a system that can generate both text and images)

At its best, an LMM can process everything under the form of tokens - without having to consider whether this token is a vision or a textual one.

# CLIP (2021) - the Pioneer
CLIP’s key contribution is its ability to map data of different modalities, text and images, into a shared embedding space. This shared multimodal embedding space makes text-to-image and image-to-text tasks so much easier.
<img src="https://huyenchip.com/assets/pics/multimodal/3-CLIP-image-classification.png">
*Zero-shot image classification with CLIP*

## Architecture
In CLIP architecture, texts & images are processed through 2 different encoders in inference time. These encoders are jointly trained together from scratch. The training goal is to maximize the similarity scores of the right (image, text) pairings while minimizing the similarity scores of the wrong pairings (contrastive learning).
<img src="https://huyenchip.com/assets/pics/multimodal/4-CLIP-architecture.png">
*CLIP's architecture*

* For the **image encoder**, the authors experimented with both ResNet and ViT. Their best-performing model is `ViT-L/14@336px`
* For the **text encoder**, CLIP uses a Transformer model similar to <a href="https://openai.com/research/better-language-models">GPT-2</a> but smaller. 

## Natural Language Supervision
For many years, image models were trained with manually annotated (image, text) datasets (e.g. ImageNet, MS COCO). This isn’t scalable. Manual annotation is time-consuming and expensive.The CLIP paper noted that none of the then-available (image, text) datasets was big and high quality enough. They created their own dataset – 400M (image, text) pairs – as follows.
1. Construct a list of 500,000 queries. Queries are common words, bigrams, and titles of popular Wikipedia articles.
2. Find images matching these queries (string and substring match). The paper mentioned this search did NOT happen on search engines but didn’t specify where. My theory is that since OpenAI already scraped the entire Internet for their GPT models, they probably just queried their internal database.
3. Each image is paired with a text that co-occurs with it (e.g. captions, comments) instead of the query since queries are too short to be descriptive.

Because some queries are more popular than others, to avoid data imbalance, they used at most 20K images for a query.

## Contrastive learning
Pre-CLIP, most vision-language models were trained using a classifier or language model objectives. Contrastive objective is a clever technique that allows CLIP to scale and generalize to multiple tasks. In contrastive objective, CLIP can learn not only which text is suitable to an image, but also which texts are not suitable as well. 
<img src="https://huyenchip.com/assets/pics/multimodal/7-clip.png">
CLIP computes the cosine similarity scores of the N^2 possible (Vi,Lj) pairings. The model is trained to maximize the similarity scores of the N correct pairings while minimizing the scores of the N^2−N incorrect pairings. For CLIP, N=32,768.

CLIP authors found that the contrastive objective provided a 12x improvement in efficiency compared to the language model objective baseline while producing higher-quality image embeddings.
<img src="https://huyenchip.com/assets/pics/multimodal/9-contrastive-learning-efficiency.png">

## Applications
Today, for many image classification tasks, CLIP is still a strong out-of-the-box baseline to be used as-is or fine-tuned. Its embeddings allow text-to-image retrieval at decent quality.
<img src="https://res.cloudinary.com/dptj6j9y9/image/upload/v1744556718/Screenshot_2025-04-13_at_10.04.07_PM_rbrlrh.png">
CLIP’s joint image-text embeddings are also useful for image generation. Given a text prompt, <a href="https://openai.com/research/dall-e">DALL-E</a> (2021) generates many different visuals and uses CLIP to rerank these visuals before showing the top visuals to users.

In 2022, OpenAI introduced <a href="https://arxiv.org/abs/2204.06125">unCLIP</a>, a text-to-image synthesis model conditioned on CLIP latents. It consists of two main components:
1. CLIP is trained and frozen. The pretrained CLIP model can generate embeddings for both text and images in the same embedding space.
2. Two things happen at image generation:
* Use CLIP to generate embedding for this text.
* Use a diffusion decoder to generate images conditioned on this embedding.
<img src="https://huyenchip.com/assets/pics/multimodal/11-unCLIP.png">

While today CLIP isn’t used directly for text generation, its image encoder is often the backbone for LMMs that can generate texts.

# Flamingo (2022) - dawn of the LMMs
Unlike CLIP, Flamingo can generate text responses. In a reductive view, Flamingo is CLIP + a language model, with added techniques to make it possible for the language model to generate text tokens conditioned on both visual and text inputs.
<img src="https://huyenchip.com/assets/pics/multimodal/12-flamingo-chatbots.png">

## Architecture
At a high level, Flamingo consists of 2 parts:
1. **Vision encoder**: a CLIP-like model is trained using contrastive learning. The text encoder of this model is then discarded. The vision encoder is frozen to be used in the main model.
2. **Language model**: Flamingo finetunes Chinchilla to generate text tokens, conditioned on visuals and text, using language model loss, with two additional components Perceiver Resampler and GATED XATTN-DENSE layers.
<img src="https://huyenchip.com/assets/pics/multimodal/13-flamingo-architecture.png">

### Dataset
Flamingo used 4 datasets: 2 (image, text) pair datasets, 1 (video, text) pair dataset, and 1 interleaved image and text dataset.
<img src="https://huyenchip.com/assets/pics/multimodal/14-flamingo-data.png">

Flamingo first trains a CLIP-like model from scratch using contrastive learning. This component only uses the 2 (image, text) pair datasets, ALIGN and LTIP, totaling 2.1B (image, text) pairs. This is 5x larger than the dataset CLIP was trained on.
* For the text encoder, Flamingo uses BERT instead of GPT-2.
* For the vision encoder, Flamingo uses a NormalizerFree ResNet (NFNet) F6 model.
* Text and vision embeddings are meanpooled before being projected to the joint embedding space.
* Flamingo uses Chinchilla as their language model. More specifically, they freeze the 9 pretrained Chinchilla LM layers. A traditional language model predicts the next text token based on the preceding text tokens. Flamingo predicts the next text token based on both the preceding text and visual tokens.
<img src="https://huyenchip.com/assets/pics/multimodal/15-lmm-text-generation.png">

To be able to generate text conditioned on both text and visual inputs, Flamingo relied on Perceiver Resampler and GATED XATTN-DENSE layers.

### Perceiver Resampler
As the visual inputs can be both images and videos, the vision encoder can produce a variable number of image or video features. Perceiver Resampler converts these variable features into a consistent 64 visual outputs.
<img src="https://huyenchip.com/assets/pics/multimodal/16-flamingo-perceiver-resampler.png">

### GATED XATTN-DENSE layers
GATED XATTN-DENSE layers are inserted between existing and frozen LM layers to allow the language model to attend more efficiently to the visual tokens when generating text tokens. Without these layers, Flamingo authors noted a drop of 4.2% in the overall score.
<img src="https://huyenchip.com/assets/pics/multimodal/17-gated%20xattn-dense.png">

# TL;DR: CLIP vs Flamingo
<img src="https://huyenchip.com/assets/pics/multimodal/18-clip-flamingo.png">

While Flamingo isn’t open-sourced, there are many open-source replications of Flamingo.
* <a href="https://huggingface.co/spaces/HuggingFaceM4/idefics_playground">IDEFICS</a> (HuggingFace)
* <a href="https://github.com/mlfoundations/open_flamingo/issues">mlfoundations/open_flamingo</a>



