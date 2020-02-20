import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import time
from PIL import Image
from collections import OrderedDict
import json

# import data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
data_transforms = {
    'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]),
                                                            
    'validation' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

    'test' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
}
# Load the datasets with ImageFolder
image_datasets = {
    'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
}
# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=False),
    'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
}

class_to_idx = image_datasets['train'].class_to_idx

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build and train your network
def get_model():
    model = models.densenet121(pretrained=True)
    return model
def build_model(hiddenLayers, class_to_idx):
    model = get_model()
    for param in model.parameters():
        param.requires_grad = False
    sizeIn = model.classifier.in_features
    print("Input size: ", sizeIn)
    sizeOut = 102
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(sizeIn, hiddenLayers)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hiddenLayers, sizeOut)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    return model

def train(model, epochs, learningRate, criterion, optimizer, trainingLoader, validationLoader):
    model.train()
    n = 40
    step = 0
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        model.cuda()
    else:
        model.cpu()
    for epoch in range(epochs):
        runningLoss = 0
        for inputs, labels in iter(trainingLoader):
            step += 1
            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                inputs = Variable(inputs)
                labels = Variable(labels) 
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.data[0]
            if step % n == 0:
                validationLoss, accuracy = validate(model, criterion, validationLoader)
                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "Training Loss: {:.3f} ".format(runningLoss/n),
                        "Validation Loss: {:.3f} ".format(validationLoss),
                        "Validation Accuracy: {:.3f}".format(accuracy))

def validate(model, criterion, dataLoader):
    model.eval()
    accuracy = 0
    testLoss = 0
    for inputs, labels in iter(dataLoader):
        if torch.cuda.is_available():
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True) 
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)
        output = model.forward(inputs)
        testLoss += criterion(output, labels).data[0]
        ps = torch.exp(output).data 
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    model.train()
    return testLoss/len(dataLoader), accuracy/len(dataLoader)

epochs = 9
learningRate = 0.001
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learningRate)
train(model, epochs, learningRate, criterion, optimizer, dataloaders['train'], dataloaders['validation'])

# Do validation on the test set
testLoss, accuracy = validate(model, criterion, dataloaders['test'])
print("Val. Accuracy: {:.3f}".format(accuracy))
print("Val. Loss: {:.3f}".format(testLoss))

:
# Save the checkpoint 
checkpointPath = 'densenet121_checkpoint.pth'

state = {
    'arch': 'densenet121',
    'learningRate': learningRate,
    'hiddenLayers': hiddenLayers,
    'epochs': epochs,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx
}

torch.save(state, checkpointPath)

def load_model(path):
    state = torch.load('densenet121_checkpoint.pth')
    learningRate = state['learningRate']
    class_to_idx = state['class_to_idx']
    model = build_model(hiddenLayers, class_to_idx)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print("Loaded '{}' (arch={}, hiddenLayers={}, epochs={})".format(
    checkpointPath, 
    state['arch'], 
    state['hiddenLayers'], 
    state['epochs']))
      
load_model('densenet121_checkpoint.pth')  
print(model)

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = 224
    # Process a PIL image for use in a PyTorch model
    width, height = image.size
    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
    resized_image = image.resize((width, height))
    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])     
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))
    return np_image_array

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

# Implement the code to predict the class from an image file
def predict(imagePath, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    useGpu = False 
    if torch.cuda.is_available():
        useGpu = True
        model = model.cuda()
    else:
        model = model.cpu()
    image = Image.open(imagePath)
    npArray = process_image(image)
    tensor = torch.from_numpy(npArray)
    if useGpu:
        varInputs = Variable(tensor.float().cuda(), volatile=True)
    else:       
        varInputs = Variable(tensor, volatile=True)
    var_inputs = varInputs.unsqueeze(0)
    output = model.forward(var_inputs)  
    ps = torch.exp(output).data.topk(topk)
    if useGpu:
        probabilities = ps[0].cpu() 
    else:
        ps[0]
    if useGpu:
        classes = ps[1].cpu() 
    else:
        ps[1]
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mappedClasses = list()
    for label in classes.numpy()[0]:
        mappedClasses.append(class_to_idx_inverted[label])
    return probabilities.numpy()[0], mappedClasses
imagePath = test_dir + '/13/image_05769.jpg'
probabilities, classes = predict(imagePath, model)
print(probabilities)
print(classes)

# Display an image along with the top 5 classes
image_path = test_dir + '/13/image_05769.jpg'
probabilitiess, classes = predict(image_path, model)
max_index = np.argmax(probabilities)
max_probability = probabilities[max_index]
label = classes[max_index]
fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)
image = Image.open(image_path)
ax1.axis('off')
ax1.set_title(cat_to_name[label])
ax1.imshow(image)
labels = []
for cl in classes:
    labels.append(cat_to_name[cl])
y_pos = np.arange(5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels)
ax2.invert_yaxis()  # probabilities read top-to-bottom
ax2.set_xlabel('Probability')
ax2.barh(y_pos, probabilities, xerr=0, align='center')
plt.show()

