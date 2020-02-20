import sys
import os
import json
import train_args
import torch
from collections import OrderedDict
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    parser = train_args.get_args()
    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s ' + __version__ + ' by ' + __author__)
    cli_args = parser.parse_args()
    if not os.path.isdir(cli_args.data_directory):
        print(f'Data directory {cli_args.data_directory} was not found.')
        exit(1)
    if not os.path.isdir(cli_args.save_dir):
        print(f'Directory {cli_args.save_dir} does not exist. Creating...')
        os.makedirs(cli_args.save_dir)
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)
    output_size = len(cat_to_name)
    print(f"Images are labeled with {output_size} categories.")

    expected_means = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    max_image_size = 224
    batch_size = 32

    training_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.25),
                                           transforms.RandomRotation(25),
                                           transforms.RandomGrayscale(p=0.02),
                                           transforms.RandomResizedCrop(max_image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(expected_means, expected_std)])

    training_dataset = datasets.ImageFolder(cli_args.data_directory, transform=training_transforms)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    if not cli_args.arch.startswith("vgg") and not cli_args.arch.startswith("densenet"):
        print("Only supporting VGG and DenseNet")
        exit(1)
    print(f"Using a pre-trained {cli_args.arch} network.")
    nn_model = models.__dict__[cli_args.arch](pretrained=True)
    densenet_input = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }
    input_size = 0
    if cli_args.arch.startswith("vgg"):
        input_size = nn_model.classifier[0].in_features
    if cli_args.arch.startswith("densenet"):
        input_size = densenet_input[cli_args.arch]
    for param in nn_model.parameters():
        param.requires_grad = False
    od = OrderedDict()
    hidden_sizes = cli_args.hidden_units
    hidden_sizes.insert(0, input_size)
    print(f"Building a {len(cli_args.hidden_units)} hidden layer classifier with inputs {cli_args.hidden_units}")
    for i in range(len(hidden_sizes) - 1):
        od['fc' + str(i + 1)] = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
        od['relu' + str(i + 1)] = nn.ReLU()
        od['dropout' + str(i + 1)] = nn.Dropout(p=0.15)
    od['output'] = nn.Linear(hidden_sizes[i + 1], output_size)
    od['softmax'] = nn.LogSoftmax(dim=1)
    classifier = nn.Sequential(od)
    nn_model.classifier = classifier
    nn_model.zero_grad()
    criterion = nn.NLLLoss()
    print(f"Setting optimizer learning rate to {cli_args.learning_rate}.")
    optimizer = optim.Adam(nn_model.classifier.parameters(), lr=cli_args.learning_rate)
    device = torch.device("cpu")
    if cli_args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU is not available. Using CPU.")
    print(f"Sending model to device {device}.")
    nn_model = nn_model.to(device)
    data_set_len = len(training_dataloader.batch_sampler)
    chk_every = 50
    print(f'Using the {device} device to train.')
    print(f'Training on {data_set_len} batches of {training_dataloader.batch_size}.')
    print(f'Displaying average loss and accuracy for epoch every {chk_every} batches.')
    for e in range(cli_args.epochs):
        e_loss = 0
        prev_chk = 0
        total = 0
        correct = 0
        print(f'\nEpoch {e+1} of {cli_args.epochs}\n----------------------------')

        for iii, (images, labels) in enumerate(training_dataloader): # VALIDATION
            inputs = Variable(images, volatile=True)
            valid_labels = Variable(labels, volatile=True)
            # Calculate loss
            output = model.forward(inputs)
            valid_loss += criterion(output, labels).data[0]
            # Calculate accuracy
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print("Training Loss: {:.3f}.. ".format(loss))
        print("Validation Loss: {:.3f}..".format(valid_loss / len(training_dataloader)))
        print("Validation Accuracy: {:.3f}..\n".format(accuracy / len(training_dataloader)))
        r_loss = 0.0
        model.train()

        for ii, (images, labels) in enumerate(training_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = nn_model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            e_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            itr = (ii + 1)
            if itr % chk_every == 0:
                print(f'avg. loss: {e_loss/itr:.4f}')
                print(f'accuracy: {(correct/total) * 100:.2f}%')
                print(f'  Batches {prev_chk:03} to {itr:03}: {avg_loss}, {acc}.')
                prev_chk = (ii + 1)
    print('Done... Saving')
    nn_model.class_to_idx = training_dataset.class_to_idx
    model_state = {
        'epoch': cli_args.epochs,
        'state_dict': nn_model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': nn_model.classifier,
        'class_to_idx': nn_model.class_to_idx,
        'arch': cli_args.arch
    }
    save_location = f'{cli_args.save_dir}/{cli_args.save_name}.pth'
    print(f"Saving checkpoint to {save_location}")
    torch.save(model_state, save_location)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
