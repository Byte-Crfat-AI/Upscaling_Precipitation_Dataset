# Super-Resolution Model using Deep Learning

This project focuses on building a Super-Resolution (SR) model using a deep learning approach. The model aims to enhance the resolution of low-resolution images by leveraging a convolutional neural network (CNN) architecture, specifically a Residual Network (ResNet) combined with UpSampling layers. The model is designed to improve the resolution of the IMD gridded dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [References](#references)

## Project Overview

The goal of this project is to enhance low-resolution images (1x1 km) to a higher resolution (0.25x0.25 km) using a Super-Resolution ResNet (SR ResNet). The project uses a dataset containing high-resolution and low-resolution images. The SR ResNet is trained to minimize the content loss between the generated high-resolution images and the actual high-resolution images.

## Model Architecture

The model is based on a Residual Network (ResNet) architecture. The core idea is to build a deep network where each layer contributes to refining the high-resolution image. Key components of the architecture include:

- **Convolutional Layers:** For extracting features from the images.
- **Residual Blocks:** For maintaining important features across layers.
- **Batch Normalization:** To stabilize and speed up training.
- **Leaky ReLU Activation:** For introducing non-linearity.
- **UpSampling Layers:** For scaling up the image resolution.
- **Final Convolutional Layer:** To produce the final high-resolution image.

### Residual Block
A residual block consists of:
- A convolutional layer followed by batch normalization and LeakyReLU activation.
- Another convolutional layer.
- Addition of the input to the output of the second convolutional layer (skip connection).

### Overall Architecture
1. Initial Convolutional Layer
2. Series of Residual Blocks
3. Upsampling Layers
4. Final Convolutional Layer

## Data Preprocessing

- The dataset contains high-resolution (HR) and low-resolution (LR) images.
- **NaN Handling:** NaN values in the dataset are replaced with 0 to ensure that the model can train without interruptions.
- **Normalization:** The images are normalized to a range of [0, 1] to help the model converge faster.

## Training

- **Loss Function:** The model is trained using a combination of Mean Squared Error (MSE) loss for pixel-wise accuracy and content loss for preserving perceptual quality.
- **Optimizer:** The Adam optimizer is used with a learning rate of `1e-4`.
- **Training Process:** The model is trained for a set number of epochs, with the learning rate being reduced on the plateau to fine-tune the model.

## Evaluation

The model is evaluated based on:
- **PSNR (Peak Signal-to-Noise Ratio):** To measure the quality of the reconstructed images.
- **SSIM (Structural Similarity Index):** To measure the similarity between the ground truth and the generated images.
- **Content Loss:** To ensure that the generated images are perceptually close to the ground truth.

## Results

The trained SR ResNet model significantly enhances the resolution of the low-resolution images. The results are evaluated using PSNR and SSIM, demonstrating the model's ability to reconstruct high-quality images.

## Dependencies

- TensorFlow
- NumPy
- Matplotlib
- Pickle (for loading datasets)
- scikit-image (for calculating PSNR and SSIM)

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/super-resolution-model.git
2.**pip install -r requirements.txt**
