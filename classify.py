import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from collections import OrderedDict
from PIL import Image
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# Define transforms for the training, validation, and testing sets
def data_transforms():
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              # transforms.Resize((224, 224)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([.5, .5, .5],
                                                                   [.5, .5, .5])])

    validation_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([.5, .5, .5],
                                                                     [.5, .5, .5])])

    testing_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([.5, .5, .5],
                                                                  [.5, .5, .5])])

    return training_transforms, validation_transforms, testing_transforms


# Load the datasets with ImageFolder
def load_datasets(train_dir, training_transforms, valid_dir, validation_transforms, test_dir, testing_transforms):
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    return training_dataset, validation_dataset, testing_dataset


# Function for processing a PIL image for use in the PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_image = Image.open(image_path)
    if pil_image.width > pil_image.height:
        factor = pil_image.width / pil_image.height
        pil_image = pil_image.resize(size=(int(round(factor * 256, 0)), 256))
    else:
        factor = pil_image.height / pil_image.width
        pil_image = pil_image.resize(size=(256, int(round(factor * 256, 0))))
    # Resize
    # if pil_image.size[0] > pil_image.size[1]:
    #    pil_image.thumbnail(size=(int(round(factor * 256, 0)), 256))
    # else:
    #    pil_image.thumbnail((224, 224))

    # Crop 
    left_margin = (pil_image.width - 224) / 2
    bottom_margin = (pil_image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Normalize
    np_image = np.array(pil_image) / 255
    mean = np.array([.5, .5, .5])
    std = np.array([.5, .5, .5])
    np_image = (np_image - mean) / std

    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))

    return np_image


# Function to convert a PyTorch tensor and display it
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([.5, .5, .5])
    std = np.array([.5, .5, .5])
    image = std * image + mean

    if title is not None:
        ax.set_title(title)

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


# Load class_to_name json file 
def load_json(json_file):
    with open(json_file, 'r') as f:
        flower_to_name = json.load(f)
        return flower_to_name


def load_eval_model(model_path):
    classifier = load_checkpoint(model_path)
    classifier.eval()
    return classifier


def predict(image_path, model, topk=3, gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        model = model.cuda()
    image = process_image(image_path)
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    output = model.forward(image)
    probabilities = torch.exp(output)
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    # print(idx_to_class)
    # print(top_indices)
    top_classes = [idx_to_class[index] for index in top_indices]

    return top_probabilities, top_classes


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['model_name'] == 'vgg':
        model = models.vgg16(pretrained=True)

    elif checkpoint['model_name'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Architecture not recognized.")

    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(
        OrderedDict([('fc1', nn.Linear(checkpoint['clf_input'], checkpoint['hidden_layer_units'])),
                     ('relu', nn.ReLU()),
                     ('drop', nn.Dropout(p=0.5)),
                     ('fc2', nn.Linear(checkpoint['hidden_layer_units'], 3)),
                     ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['model_state_dict'])

    return model


# Function to display an image along with the top 5 classes
def display_image(image_dir, flower_to_name, classes, probabilities):
    # Plot flower input image
    plt.figure(figsize=(6, 10))
    plot_1 = plt.subplot(2, 1, 1)

    image = process_image(image_dir)

    key = image_dir.split('/')[-2]

    flower_title = flower_to_name[key]

    imshow(image, plot_1, title=flower_title)

    # Convert from the class integer encoding to actual flower names
    flower_names = [flower_to_name[i] for i in classes]

    # Plot the probabilities for the top 5 classes as a bar graph
    plt.subplot(2, 1, 2)

    sb.barplot(x=probabilities, y=flower_names, color=sb.color_palette()[0])

    plt.show()
