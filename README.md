# Lung Cancer Detection using Convolutional Neural Network and OpenAI

This project focuses on the development of a lung cancer detection system using Convolutional Neural Networks (CNN) and OpenAI. The goal is to build an image classification model that can identify lung cancer from histopathological images. Additionally, it integrates OpenAI's GPT-3 for generating image descriptions based on the predicted class labels.

## Motivation

Lung cancer is a major health concern globally, and early detection plays a crucial role in improving patient outcomes. Histopathological image analysis is a powerful tool in the early diagnosis of lung cancer. By developing a deep learning-based system, we aim to assist pathologists in identifying malignant tissue patterns more accurately and quickly.

## Features

- Utilizes state-of-the-art deep learning techniques for image classification.
- Leverages data augmentation to enhance model robustness.
- Integrates OpenAI's GPT-3 to generate detailed descriptions of image findings.
- Aims to improve the accuracy of lung cancer detection and provide interpretability through generated text descriptions.

## About the Dataset

This dataset contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format.

The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the Augmentor package.

There are five classes in the dataset, each with 5,000 images, being:

- Lung benign tissue
- Lung adenocarcinoma
- Lung squamous cell carcinoma
- Colon adenocarcinoma
- Colon benign tissue
  
  ![image](https://github.com/Jsujanchowdary/-Lung-Cancer-Detection-using-Convolutional-Neural-Network-and-Open-Ai/assets/91127394/9a3aa993-cc59-4a9a-9cd8-642fe00f7c11)
  ![image](https://github.com/Jsujanchowdary/-Lung-Cancer-Detection-using-Convolutional-Neural-Network-and-Open-Ai/assets/91127394/0d50b230-4600-4432-bf7f-b611229789d6)
  ![image](https://github.com/Jsujanchowdary/-Lung-Cancer-Detection-using-Convolutional-Neural-Network-and-Open-Ai/assets/91127394/0f2cb015-6aaf-4e17-ac3b-a810c92b6f49)




**Dataset Link**: [View Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/)

## Prerequisites

Before you can run the code, you need to ensure that you have the necessary libraries and an API key:

- **numpy** (as **np**): A library for numerical operations and array manipulation.
- **pandas** (as **pd**): A library for data manipulation and analysis.
- **matplotlib.pyplot** (as **plt**): Used for data visualization, including creating plots and charts.
- **PIL** (Python Imaging Library): A library for working with image files.
- **glob**: Used for file path manipulation and searching for files in directories.
- **sklearn**: The scikit-learn library for machine learning, which provides various tools for classification, regression, clustering, and more.
- **cv2** (OpenCV): An open-source computer vision library used for image processing.
- **gc** (garbage collector): Python's built-in module for controlling the automatic garbage collection.
- **tensorflow** (as **tf**): A popular open-source machine learning framework developed by Google.
- **keras**: An integrated deep learning framework that comes with TensorFlow and is used for building and training neural networks.
- **openai**: The OpenAI library is required for GPT-3 integration. You must have an API key from OpenAI to use it.

## Usage

Here's how the code works step by step:

1. **Define the Data Path and Directory Structure**:
   - The code begins by specifying the path to a dataset stored in a ZIP file. The `ZipFile` module is used to extract the dataset, which is organized into a directory structure.

2. **Data Preprocessing and Augmentation**:
   - The script sets parameters for image preprocessing and augmentation using `ImageDataGenerator`. This includes rescaling pixel values to the range [0, 1] and defining data augmentation techniques like rotation, shifting, and flipping to increase the diversity of the training data.

3. **Loading and Preprocessing Images**:
   - The code iterates through the subdirectories in the dataset directory and reads images using OpenCV (`cv2`). The images are resized to a specified size (defined by `IMG_SIZE`) and added to the `X` list. The corresponding labels (class indices) are added to the `Y` list.

4. **One-Hot Encoding**:
   - The labels (`Y`) are one-hot encoded using `pd.get_dummies`. This converts categorical labels into binary vectors.

5. **Train-Test Split**:
   - The dataset is split into training and validation sets using `train_test_split` from scikit-learn.

6. **Building a Convolutional Neural Network (CNN)**:
   - A CNN model is defined using Keras. The model consists of convolutional layers, max-pooling layers, fully connected layers, and batch normalization layers. It is compiled with an optimizer, loss function, and evaluation metric.

7. **Model Training and Callbacks**:
   - The model is trained using the training data. Training is monitored using callbacks like early stopping (`EarlyStopping`) and learning rate reduction (`ReduceLROnPlateau`) to improve training efficiency.

8. **Data Visualization**:
   - The code uses `matplotlib` to plot the training and validation loss and accuracy over epochs.

     ![image](https://github.com/Jsujanchowdary/-Lung-Cancer-Detection-using-Convolutional-Neural-Network-and-Open-Ai/assets/91127394/c6f57cef-a10b-4d34-a794-51b385bdb9ec)


9. **Model Evaluation**:
   - The model is evaluated on the validation set using metrics like the confusion matrix and classification report from `sklearn`.
  
     ![image](https://github.com/Jsujanchowdary/-Lung-Cancer-Detection-using-Convolutional-Neural-Network-and-Open-Ai/assets/91127394/b5c21095-998b-46ec-be22-00d638f06432)


10. **GPT-3 Integration (Additional)**:
    - The code integrates OpenAI's GPT-3 API for generating image descriptions based on prompts. It defines a function `generate_image_descriptions` that sends image prompts to GPT-3 and receives generated descriptions.
   
      ![image](https://github.com/Jsujanchowdary/-Lung-Cancer-Detection-using-Convolutional-Neural-Network-and-Open-Ai/assets/91127394/ac79e431-c811-44b3-81d3-236e865e8f37)



## Acknowledgments

- We would like to express our gratitude to the creators of the dataset for making it publicly available.
- The OpenAI team for providing access to the GPT-3 API, enabling advanced image description generation.

Thank you for being a part of our lung cancer detection initiative. Together, we can make a positive impact on healthcare.
