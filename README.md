# Brain Tumor Detection using Convolutional Neural Network (CNN)

This project focuses on detecting brain tumors from MRI images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on a dataset consisting of images categorized into four classes: Glioma, Meningioma, NoTumor, and Pituitary.

## Table of Contents
- [Introduction](#introduction)
- [Approach](#approach)
- [Categories](#categories)
- [CNN Model](#cnn-model)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Prediction](#prediction)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Introduction

Brain tumors are abnormal growths of cells in the brain that can be life-threatening. Accurate detection from MRI images is crucial for timely and effective treatment. This project aims to classify MRI images into four categories:
- Glioma
- Meningioma
- NoTumor
- Pituitary

## Approach

We use a Convolutional Neural Network (CNN) due to its effectiveness in handling visual data. The CNN model is trained to learn the intricate features of MRI images and classify them into the aforementioned categories.

## Categories

- **Glioma**: A type of tumor that occurs in the brain and spinal cord.
- **Meningioma**: A tumor that arises from the meninges, the membranes that surround the brain and spinal cord.
- **NoTumor**: MRI images that do not show any signs of a tumor.
- **Pituitary**: Tumors that occur in the pituitary gland, which is located at the base of the brain.

## CNN Model

The CNN model architecture includes:
1. **Data Augmentation**: Increases the diversity of the training dataset.
2. **Rescaling**: Normalizes pixel values to the range [0, 1].
3. **Convolutional Layers**: Extracts features using filters.
4. **MaxPooling Layers**: Reduces the spatial dimensions of the feature maps.
5. **Dropout Layer**: Prevents overfitting.
6. **Flatten Layer**: Converts 2D matrices to 1D vectors.
7. **Dense Layers**: Performs classification based on the extracted features.

### Model Architecture


The CNN model is built using TensorFlow and Keras. The model architecture includes multiple convolutional layers followed by max-pooling layers and dense layers for classification.

The model is defined and trained twice, utilizing different techniques to enhance accuracy and performance:

1. **Initial Model Training**: The initial model is defined with multiple convolutional layers, max-pooling layers, and dense layers. It is compiled and trained on the dataset to get a baseline accuracy.

2. **Enhanced Model Training**: A second model is then defined and trained using advanced data augmentation techniques (such as random flipping, rotation, and zoom) and hyperparameter tuning. This step aims to improve the initial model's accuracy and robustness.


## Data Preprocessing

The dataset is split into training and validation sets with an 80-20 ratio. Data augmentation techniques such as random flipping, rotation, and zoom are applied to the training data to improve the model's generalization capability.

## Training the Model

1. Load and preprocess the dataset.
2. Split the dataset into training and validation sets.
3. Define the initial CNN model architecture.
4. Compile and train the initial model using the Adam optimizer and Sparse Categorical Crossentropy loss function.
5. Define and train the enhanced model with improved data augmentation and hyperparameters.

## Evaluating the Model

The model's performance is evaluated using training and validation accuracy and loss metrics. Plots of accuracy and loss over epochs are created to visualize the training process.

## Prediction

The trained model is used to make predictions on new MRI images. The model outputs the predicted class along with confidence scores.

## Results

- The initial model achieved an accuracy of `97.85%` on the training set and `93.45%` on the validation set.
- The enhanced model achieved an accuracy of `98.53%` on the training set and `95.63%` on the validation set after `100` epochs.
- The model demonstrated high confidence and accuracy in classifying MRI images into the correct categories.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. Mount Google Drive in Colab:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. Install the required libraries:
    ```bash
    pip install tensorflow
    pip install matplotlib
    ```

## Usage

1. Load and preprocess the MRI image dataset.
2. Train the CNN model on the dataset.
3. Evaluate the model's performance.
4. Use the model to make predictions on new MRI images.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

