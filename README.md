# Integrated Deepfake Detection System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Project Structure](#project-structure)
4. [Features](#features)
5. [Data Used](#data-used)
6. [Model Architecture](#model-architecture)
    - [Spatial Feature Extraction](#spatial-feature-extraction)
    - [Temporal Feature Extraction](#temporal-feature-extraction)
    - [Micro-Expression Analysis](#micro-expression-analysis)
    - [Feature Fusion Layer](#feature-fusion-layer)

## Project Overview

The Integrated Deepfake Detection System is a comprehensive project aimed at detecting deepfake videos by analyzing spatial, temporal, and micro-expression features. The system utilizes state-of-the-art deep learning models, including Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), to extract and fuse features for accurate deepfake detection.

This project is developed as part of a submission to SIH 2024 (Smart India Hackathon) and aims to push the boundaries of automated deepfake detection techniques.

## Motivation

With the rise of deepfake technology, the ability to manipulate video content has become increasingly sophisticated, posing serious threats to privacy, security, and authenticity. This project addresses the need for robust deepfake detection mechanisms that can identify manipulated content by analyzing various facets of video data.

## Project Structure

The project is structured into several key components:

- **Data Preprocessing**: Prepares video frames for feature extraction.
- **Spatial Feature Extraction**: Uses pre-trained CNN models to extract spatial features.
- **Temporal Feature Extraction**: Utilizes BiLSTM networks to capture temporal dependencies across video frames.
- **Micro-Expression Analysis**: Analyzes subtle facial movements to detect inconsistencies indicative of deepfakes.
- **Feature Fusion Layer**: Integrates spatial, temporal, and micro-expression features for final decision-making.
- **Output**: Generates a report indicating whether the video is a deepfake or genuine.

## Features

- **Multi-Feature Detection**: Combines spatial, temporal, and micro-expression features to enhance detection accuracy.
- **Real-Time Analysis**: Capable of analyzing videos in real time for practical applications.
- **Attention Mechanism**: Implements attention layers to focus on critical aspects of the input data.
- **Scalability**: Designed to be scalable and adaptable to different datasets and video lengths.

## Data Used

The system is trained and tested using the **FaceForensics++** dataset, which contains both original and manipulated video sequences. The dataset is organized into folders for 'original sequences' and 'manipulated sequences', providing a rich source of data for training and evaluation.

## Model Architecture

### Spatial Feature Extraction

- **Model Used**: ResNet50 and VGG16
- **Purpose**: Extract spatial features from each frame of the video, capturing detailed facial features.
- **Implementation**: Utilizes pre-trained ResNet50 and VGG16 models, with additional custom layers to refine feature extraction.

### Temporal Feature Extraction

- **Model Used**: BiLSTM (Bidirectional Long Short-Term Memory)
- **Purpose**: Capture temporal dependencies across frames, analyzing how features change over time.
- **Implementation**: A sequence of feature vectors is fed into BiLSTM layers, followed by attention mechanisms to focus on significant temporal patterns.

### Micro-Expression Analysis

- **Model Used**: Custom CNN
- **Purpose**: Detect subtle micro-expressions that are hard to manipulate in deepfake videos.
- **Implementation**: A dedicated CNN model extracts fine-grained facial movements, which are then analyzed for inconsistencies.

### Feature Fusion Layer

- **Purpose**: Integrate spatial, temporal, and micro-expression features to form a comprehensive feature set.
- **Implementation**: Features from different modules are concatenated and processed through dense layers with attention mechanisms for final classification.
