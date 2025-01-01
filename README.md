 # AI-organs-classifier
## Introduction

The AI Classifier App is a graphical interface application designed to identify specific body parts in medical images. It employs deep learning techniques to train a robust model that classifies images into predefined categories. This project aims to enhance medical imaging analysis and aid in diagnostic workflows.


###### image of program:
<div>
  <img src ="https://github.com/user-attachments/assets/f1968ef5-57a3-4ad7-a58d-702573e5ff21" >
</div>

## Features

- GUI Interface: User-friendly interface for training, testing, and managing the model.

- Deep Learning Integration: Utilizes PyTorch and ResNet18 for model training and inference.

- Image Preprocessing: Includes advanced preprocessing pipelines for training and testing.

- Customizable Dataset: Supports dynamic addition of labeled images for training.

- Real-time Feedback: Displays training progress and prediction results in real time.

- Logging and Persistence: Saves training data and model checkpoints to disk for future use.

## Requirements

- Python 3.8 or higher

- A CUDA-compatible GPU is recommended for faster training.

- Required libraries (listed in [requirement.txt](https://github.com/ziad0nassif/AI-organs-classifier/blob/89743bc4f790a4248fb46f7a28da17428fc17a09/requirements.txt) )

## Logging

The application logs key information including:

- Training progress and accuracy.

- Prediction results with probabilities.

- Errors encountered during image loading or model training.

- Logs are displayed in the GUI and saved in structured formats for analysis.

## Dataset Structure

- The application uses a dataset divided into labeled categories, with each category corresponding to a body part. Data is dynamically managed through the GUI and saved in JSON format for persistence.


## Contributing

- We welcome contributions! To contribute:
