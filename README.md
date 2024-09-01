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
7. [Installation](#installation)
8. [Usage](#usage)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)

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

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/deepfake-detection-system.git
    cd deepfake-detection-system
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the FaceForensics++ dataset** and place it in the `data/` directory.

## Results

The Integrated Deepfake Detection System has shown promising results in detecting deepfake videos with high accuracy. The system's ability to analyze spatial, temporal, and micro-expression features enables it to outperform traditional detection methods. Detailed results, including precision, recall, and F1-score, can be found in the `results/` directory.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact

- **Name**: Your Name
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/your-username/deepfake-detection-system](https://github.com/your-username/deepfake-detection-system)

Feel free to reach out if you have any questions or suggestions!
