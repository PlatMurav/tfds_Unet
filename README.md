# Segmentation Model with TensorFlow and TFRecords
## Overview
This repository contains code for building a semantic segmentation model using TensorFlow and TensorFlow Datasets.
<br>The model architecture used in this repository is a Convolutional Neural Network (CNN) based on U-Net and designed for semantic segmentation tasks. It takes input images and outputs segmentation masks with class labels for each pixel.

### Dataset
The datasets used for training, validation and testing the model consist of images and their corresponding segmentation masks.
* The images are stored in a folder ("data/images")
* The annotations (JSON-file) are provided in COCO format ("data/instances.json").

## Folder structure
```
dataset
│
my_project/
│
├── data/
│ ├── instances.json
│ └── images/
│   ├── camera_0.png
│   ├── camera_1.png
│   └── ...
├── my_custom_dataset/
│ ├── init.py
│ ├── my_custom_dataset.py
│ └── my_custom_dataset_test.py
├── main.py
├── model.py
└── unit_tests.py
```
### Files Description
- `data/`: Directory containing images and json-file.
  - `images`: Directory containing all necessary images.
  - `instances.json`: json-file in coco format that describes the data.
- `my_custom_dataset/`: Directory containing dataset definition files.
  - `__init__.py`: Makes the directory a Python package.
  - `my_custom_dataset.py`: Contains the `MyCustomDataset` class definition to create and load the dataset.
- `main.py`: Script to load datasets, train the model and evaluate it.
- `unit_tests.py`: Script with unit tests for our model.
- `model.py`: Defines the U-Net model architecture and the class for handling our datasets.


## Requirements
To be albe to use the model yuo should have:
* Images
* JSON-file in coco-format

A folder with images that may contain some of the 10 objects and JSON-file (in coco-format) with annotations for the images.

All necessary **dependencies** You can install using pip:
```python
pip install -r requirements.txt
```

## Usage
*based on main.py*
### Creating a new model
If you want to train a new model you simply need to copy steps that are shown in main.py.
<br>At first, you need:
1. import our class
2. create an instance
```python
from model import Segmentator

model = Segmentator()
```
Now "pre-model" is created and we can start using it. 

### Creating Datasets
We have our own dataset we need to create datasets so we could start training.
<br>To create all necessary datasets we need call **.load_dataset()** method and pass one of the three arguments: "train", "val", "test":
```python
# Defining datasets
train_dataset = model.load_dataset('train')
val_dataset = model.load_dataset('val')
```

### Initializing the model
Now we can Initialize our model by writing:
```python
model.build_model()
```
This method creates CNN-structure with 6 convolutional layers internally and sets Adam optimizer.

### Training and evalation
To start trainig the model we call **.build_model()** method and pass training and validation datasets which were created earlier:
```python
model.train_model(train_dataset, val_dataset)
```
**Notes**:
* The model is trained using the TensorFlow framework: loads the dataset from TFRecord files, preprocesses the data and trains the CNN model using optimization.

### Saving and loading the model
When our model is ready we can save it providing path to the method:
```python
#native Keras format
model.save_model('trained_model_t800.keras')
```
