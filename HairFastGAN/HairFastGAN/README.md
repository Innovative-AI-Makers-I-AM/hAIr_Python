---
license: mit
pipeline_tag: image-to-image
---

# HairFastGAN: Realistic and Robust Hair Transfer with a Fast Encoder-Based Approach

<a href="https://arxiv.org/abs/2404.01094"><img src="https://img.shields.io/badge/arXiv-2404.01094-b31b1b.svg" height=22.5></a>
<a href="https://huggingface.co/spaces/AIRI-Institute/HairFastGAN"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg" height=22.5></a>
<a href="https://github.com/AIRI-Institute/HairFastGAN"><img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" height=22.5></a>
<a href="https://colab.research.google.com/#fileId=https://huggingface.co/AIRI-Institute/HairFastGAN/blob/main/notebooks/HairFast_inference.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a>



> Our paper addresses the complex task of transferring a hairstyle from a reference image to an input photo for virtual hair try-on. This task is challenging due to the need to adapt to various photo poses, the sensitivity of hairstyles, and the lack of objective metrics. The current state of the art hairstyle transfer methods use an optimization process for different parts of the approach, making them inexcusably slow. At the same time, faster encoder-based models are of very low quality because they either operate in StyleGAN's W+ space or use other low-dimensional image generators. Additionally, both approaches have a problem with hairstyle transfer when the source pose is very different from the target pose, because they either don't consider the pose at all or deal with it inefficiently. In our paper, we present the HairFast model, which uniquely solves these problems and achieves high resolution, near real-time performance, and superior reconstruction compared to optimization problem-based methods. Our solution includes a new architecture operating in the FS latent space of StyleGAN, an enhanced inpainting approach, and improved encoders for better alignment, color transfer, and a new encoder for post-processing. The effectiveness of our approach is demonstrated on realism metrics after random hairstyle transfer and reconstruction when the original hairstyle is transferred. In the most difficult scenario of transferring both shape and color of a hairstyle from different images, our method performs in less than a second on the Nvidia V100.
> 

<p align="center">
  <img src="docs/assets/logo.webp" alt="Teaser"/>
  <br>
The proposed HairFast framework allows to edit a hairstyle on an arbitrary photo based on an example from other photos. Here we have an example of how the method works by transferring a hairstyle from one photo and a hair color from another.
</p>

This repository contains the pretrained weights for our method, the inference and training code can be found on GitHub: ![https://github.com/AIRI-Institute/HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN)
