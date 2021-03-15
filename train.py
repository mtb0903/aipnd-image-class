from collections import OrderedDict
from time import perf_counter
from pathlib import Path
from datetime import datetime
import json
import argparse
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from workspace_utils import keep_awake


def path_exists(path, replace_keyword="", is_file=False):
    """
    Validates the existence of an file or directory path and returns it

    :param path: Path to be valided
    :param replace_keyword: Positional argument keyword which will be replaced, if required
    :param is_file: True, if path should be an existing file; False, if it should be a directory
    :return pathlib.Path: path to file or directory, if existent
    :raise Exception: In case file or directory path does not exist
    """
    p = Path(path.replace(replace_keyword, "")).resolve()

    if is_file and p.is_file():
        return p
    elif not is_file and p.is_dir():
        return p
    else:
        if is_file:
            message = "File path does not exist or is not a file."
        else:
            message = "Directory path does not exist or is not a directory."
        raise Exception(message)


def get_args():
    """
    Validates and returns parsed CLI arguments as dictionary

    :return dictionary: parsed CLI arguments
    """
    # Default root directory will be the directory in which this file is located
    default_root_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='Train, validate, test and save image classifier model')
    parser.add_argument('data_dir', type=lambda p: path_exists(p, replace_keyword="data_dir="),
                        metavar='<path to dir>', nargs='?', default=default_root_dir.joinpath('flowers'),
                        help='directory containing a train, valid and test directory, which each contain image files '
                             'as expected by torchvision.datasets.ImageFolder; defaults to <script_dir>/flowers')
    parser.add_argument('-s', '--save_dir', type=lambda p: path_exists(p), metavar='<path to dir>', nargs='?',
                        default=default_root_dir, help='directory to which the torch checkpoint '
                                                       'of the trained model will be saved')
    parser.add_argument('-a', '--arch', type=str, metavar='<model_architecture>', nargs='?', default='vgg16',
                        help='image classification model architecture; currently supported is '
                             'torchvision.models.vgg16, torchvision.models.densenet161 and '
                             'torchvision.models.resnet18; defaults to vgg16')
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='<learning rate float>', nargs='?',
                        default=0.001, help='learning rate; defaults to 0.001')
    parser.add_argument('-e', '--epochs', type=int, metavar='<integer>', nargs='?', default=5,
                        help='number of training / validation epochs; defaults to 5')
    parser.add_argument('-hn', '--hidden_units', type=int, metavar='<integer>', nargs='*', default=[512],
                        help='number of neurons per hidden layer of image classifier; example value for '
                             '3 hidden layers: 2048 1024 512; defaults to 1 layer with 512 neurons')
    parser.add_argument('-g', '--gpu', action='store_true', help='use gpu, if available; defaults to False')
    parser.add_argument('-b', '--batch_size', type=int, metavar='<integer>', nargs='?', default=32,
                        help='torch.utils.data.DataLoader batch_size; defaults to 32')
    parser.add_argument('-dp', '--dropout_p', type=float, metavar='<float>', nargs='?', default=0.5,
                        help='Probability for classifier neuron dropouts during training; defaults to 0.5')
    parser.add_argument('-c', '--category_names', type=lambda p: path_exists(p, is_file=True), metavar='<path to file>', nargs='?',
                        default=default_root_dir.joinpath('cat_to_name.json'),
                        help='JSON file containing a mapping of categories and label names for targets'
                             'defaults to cat_to_name.json')

    try:
        # vars returns dictionary
        return vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))


def get_data_loaders(parameters):
    """
    Use torchvision.transforms and torchvision.datasets.ImageFolder to create and return torch.utils.data.DataLoader
    for training, validation and testing. Additionally, the number of unique classes is returned

    :param parameters: dictionary that provides root data directory and DataLoader batch size
    :return int, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader:
    number of unique classes, train data loader, validation data loader, test data loader
    """

    data_dir = parameters.get('data_dir', 'flowers')
    batch_size = parameters.get('batch_size', 32)

    # Data directories
    train_dir = data_dir.joinpath('train')
    valid_dir = data_dir.joinpath('valid')
    test_dir = data_dir.joinpath('test')

    # Transforms for training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(120),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])

    # Definition of training, validation and testing datasets using ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Store mapping between classes and indices from dataset
    parameters['train_datasets_class_to_idx'] = train_datasets.class_to_idx

    # Store number of records per dataset
    parameters['no_records_train'] = len(train_datasets)
    parameters['no_records_valid'] = len(valid_datasets)
    parameters['no_records_test'] = len(test_datasets)

    # Classes = targets = labels
    no_unique_train_labels = len(train_datasets.classes)
    # Alternative in newer torchvision versions:
    # no_unique_train_labels = len(set(train_datasets.targets))

    # Loading the training, validation and testing datasets as generator using DataLoader
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)

    return no_unique_train_labels, train_loader, valid_loader, test_loader


def get_cat_to_name_mapping(file):
    """
    Read the json file provided and return a dictionary containing the mapping between categories and label names

    :param file: path to JSON file containing mapping between categories and label names
    :return cat_to_name: dictionary containing mapping between categories and label names
    """
    try:
        with open(file, 'r') as f:
            # Get mapping between categories and labels as dictionary
            cat_to_name = json.load(f)
    except Exception as e:
        print("WARN: {} could no be loaded as dictionary. Will return empty dictionary. Exception: {}".format(file, e))
        # Create empty dictionary
        cat_to_name = {}

    return cat_to_name


def create_model(parameters):
    """
    Create a neural network based on a pretrained network architecture with
    a custom classifier.

    :param parameters: network parameters
    :return: model: pretrained model with custom classifier
    """

    # Get pre-trained network
    arch = parameters.get('arch', 'vgg16')
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = parameters.get('input_size', 25088)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = parameters.get('input_size', 512)
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_size = parameters.get('input_size', 2208)

    # Ensure that gradients of pre-trained model are not considered in back propagation
    for param in model.parameters():
        param.requires_grad = False

    # Note: Per default gradients of new layer added are enabled (requires_grad=True)

    # Classifier neural network layer sizes
    hidden_units = parameters.get('hidden_units', 0)
    output_size = parameters.get('output_size', 0)

    if hidden_units == 0:
        raise Exception("ERROR: model not created. 'hidden_units' not defined")
    if output_size == 0:
        raise Exception("ERROR: model not created. 'output_size' not defined")

    # Dropout probability
    dropout_p = parameters.get('dropout_p', 0)
    if dropout_p == 0:
        print("WARN: 'dropout_p' not defined, set dropout probability to default: 0.5")
        dropout_p = 0.5

    # Initialize ordered dict with input layer for the new classifier
    layer_counter = 1
    seq_dict = OrderedDict([
        ('fc{}'.format(layer_counter), nn.Linear(input_size, hidden_units[0])),
        ('relu{}'.format(layer_counter), nn.ReLU()),
        ('dropout{}'.format(layer_counter), nn.Dropout(p=dropout_p))])

    # Update ordered dict depending on hidden_units
    for i in range(len(hidden_units)):
        layer_counter += 1
        if i + 1 < len(hidden_units):
            seq_dict['fc{}'.format(layer_counter)] = nn.Linear(hidden_units[i], hidden_units[i+1])
            seq_dict['relu{}'.format(layer_counter)] = nn.ReLU()
            seq_dict['dropout{}'.format(layer_counter)] = nn.Dropout(p=dropout_p)
        else:
            # Add output layer
            seq_dict['fc{}'.format(layer_counter)] = nn.Linear(hidden_units[i], output_size)

    # Add LogSoftmax to output layer
    seq_dict['out_log_softmax'] = nn.LogSoftmax(dim=1)

    # Create classifier
    classifier = nn.Sequential(seq_dict)

    # Overwrite classifier of pre-trained model
    if arch == 'resnet18':
        model.fc = classifier
    elif arch == 'vgg16' or arch == 'densenet161':
        model.classifier = classifier

    # Store mapping between classes and indices from dataset in model
    model.class_to_idx = parameters.get('train_datasets_class_to_idx')

    return model


def get_device(request_gpu):
    """
    Return torch.device depending on request_gpu and GPU availability

    :param request_gpu: Set to True for requesting use of GPU
    :return torch.device:
    """
    if request_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == torch.device("cpu"):
            print("WARN: GPU requested, but is not available. Use CPU instead.")
    else:
        device = torch.device("cpu")

    return device


def train_image_classifier(parameters, device, model, criterion, optimizer, train_loader, valid_loader,
                           train_losses, valid_losses, valid_accuracies):
    """
    Train and validate image classification model

    :param parameters: dictionary providing model parameters
    :param device: torch.device indicating if GPU or CPU will be used
    :param model: the neural network model
    :param criterion: the loss / error function
    :param optimizer: the optimizer for backpropagation
    :param train_loader: training torch.utils.data.DataLoader
    :param valid_loader: validation torch.utils.data.DataLoader
    :param train_losses: list to store training loss calculated at a certain interval
    :param valid_losses: list to store validation loss calculated at a certain interval
    :param valid_accuracies: list to store validation accuracy calculated at a certain interval
    :return:
    """

    # Initialize parameters
    train_loss_accuracy_batch_interval = parameters.get('train_loss_accuracy_batch_interval', 5)
    epochs = parameters.get('epochs', 5)

    # Training loop
    # keep_awake is provided by Udacity to ensure that anything that happens inside this loop will keep the workspace active
    # for e in range(epochs):
    for e in keep_awake(range(epochs)):

        # Enable training mode, which uses dropouts
        model.train()

        print("***************************")
        print("Epoch {}/{}".format(e+1, epochs))

        running_training_loss = 0
        batch_count = 0
        train_image_count = 0
        for train_images, train_labels in train_loader:

            # Count images processed by training
            train_image_count += len(train_images)

            # Count batches used for training
            batch_count += 1

            # Move input and label tensors to GPU (if available) or CPU
            train_images, train_labels = train_images.to(device), train_labels.to(device)

            # Ensure that the gradient is not accumulated with each iteration
            optimizer.zero_grad()

            # Forward feeding the model
            train_output = model(train_images)

            # Calculate the loss (error function)
            train_loss = criterion(train_output, train_labels)

            # Backpropagation
            train_loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss for each trained image
            running_training_loss += train_loss.item()

            # Validate the model after each loss_accuracy_batch_interval
            if batch_count % train_loss_accuracy_batch_interval == 0:

                # Validate the model
                test_image_classifier(device, model, criterion, valid_loader, valid_losses, valid_accuracies, False)

                # Calculate loss for batch used for last training
                train_losses.append(running_training_loss / batch_count)

                print("----")
                print("Processed {} training batches with {} processed images".format(batch_count, train_image_count))
                print("Training loss: {}".format(train_losses[-1]))
                print("Validation loss: {}".format(valid_losses[-1]))
                print("Validation accuracy: {}".format(valid_accuracies[-1]))


def test_image_classifier(device, model, criterion, data_loader, losses, accuracies, calc_per_batch):
    """
    Validate / test image classification model

    :param device: torch.device indicating if GPU or CPU will be used
    :param model: the neural network model
    :param criterion: the loss / error function
    :param data_loader: data provided via torch.utils.data.DataLoader
    :param losses: list to store loss calculated at a certain interval
    :param accuracies: list to store accuracy calculated at a certain interval
    :param calc_per_batch: True, if you want to process loss and accuracy calculation after each batch
    :return:
    """

    # Enable validation / testing mode, which disables dropouts
    model.eval()

    # Initialize variables for calculation per batch
    running_loss = 0
    running_accuracies = 0
    image_count = 0
    batch_count = 0

    # For validation / testing no new gradient calculation is required, backpropagation will not be used
    with torch.no_grad():

        # Iterate through test data per batch
        for images, labels in data_loader:

            # Count images processed by training
            image_count += len(images)

            # Count batches used for training
            batch_count += 1

            # Loss and accuracy per batch
            batch_loss = 0
            batch_accuracy = 0

            # Move input and label tensors to GPU (if available) or CPU
            images, labels = images.to(device), labels.to(device)

            # Forward feeding the model to validate the input
            output = model(images)

            # Calculate the loss (error function)
            loss = criterion(output, labels)

            # Accumulate loss for each validated image
            running_loss += loss.item()

            # Calculate out probability, because logsoftmax is used
            ps = torch.exp(output)

            # Get the best probability value and it's index
            top_p, top_k = ps.topk(1, dim=1)

            # Create true / false tensor with matches
            # Create labels tensor 2D view
            equals = top_k == labels.view(*top_k.shape)

            # Alternative: running_accuracies += len(equals[equals == True]) / len(equals)
            running_accuracies += torch.mean(equals.type(torch.FloatTensor)).item()

            # Process loss and accuracy per batch
            if calc_per_batch:
                batch_loss += loss.item()
                batch_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print("----")
                print("Batch {}/{} - Loss: {}".format(batch_count, len(data_loader), batch_loss))
                print("Batch {}/{} - Accuracy: {}".format(batch_count, len(data_loader), batch_accuracy))
                losses.append(batch_loss)
                accuracies.append(batch_accuracy)

        else:
            if calc_per_batch:
                print("----")
                print("Total processed batches: {}".format(len(data_loader)))
                print("Total processed images: {}".format(image_count))
                print("Average loss: {}".format(running_loss / len(data_loader)))
                print("Average accuracy: {}".format(running_accuracies / len(data_loader)))
            else:
                accuracies.append(running_accuracies / len(data_loader))
                losses.append(running_loss / len(data_loader))


def print_parameters(parameters):
    """
    Print network parameters and training results

    :param parameters: dictionary containing network parameters and training, validation and test results
    :return:
    """

    print("Model parameters:")

    state_dict_exists = False

    for k, v in parameters.items():
        if k in ['train_losses', 'valid_losses', 'test_losses', 'valid_accuracies', 'test_accuracies']:
            print("{}: {} (first 5 values)".format(k, v[:5]))
        elif k == 'state_dict':
            print("'state_dict' key exists, but won't be printed")
            state_dict_exists = True
        else:
            print("{}: {}".format(k, v))

    if not state_dict_exists:
        print("'state_dict' key does not exist")


def save_checkpoint(model, parameters):
    """
    Save parameters dictionary containing model parameters and model as pytorch checkpoint file to disk

    :param model: neural network model
    :param parameters: dictionary containing model parameters for reconstruction and
    training, validation and test results as reference
    :return:
    """

    # Store in parameters the state dict (weights and biases)
    parameters['state_dict'] = model.state_dict()

    # Get directory and timestamp
    save_dir_path = parameters.get('save_dir', Path(__file__).parent)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Define path of checkpoint file
    checkpoint_file_path = save_dir_path.joinpath('{}_{}_checkpoint.pth'.format(timestamp,
                                                                                parameters.get('arch', 'vgg16')))

    # Save model to checkpoint file
    torch.save(parameters, checkpoint_file_path)

    print('Checkpoint saved to {}'.format(checkpoint_file_path))


def get_day_hour_min_sec_str(duration_in_s):
    """
    Return a string representing the duration in days, hours, minutes and seconds of the input value duration_in_s

    :param duration_in_s: Duration in seconds
    :return:
    """
    return "{:d}d {:d}h {:d}m {:d}s".format(int(duration_in_s / (24 * 3600)), int(duration_in_s / 3600),
                                            int(duration_in_s / 60), int(duration_in_s % 60))


def main():
    """
    Main procedure

    :return:
    """
    # Define start time
    start_time_main = perf_counter()

    # Get CLI arguments
    print('Get CLI arguments ...')
    parameters = get_args()

    # Load datasets and get data loaders for train, validation and test purposes
    # and get number of unique train dataset targets to determine output layer size
    print('Load data ...')
    no_unique_train_labels, train_loader, valid_loader, test_loader = get_data_loaders(parameters)

    # Load dictionary containing the mapping of classification categories and label names
    print('Load mapping of categories to label names ...')
    cat_to_name_file = parameters.get('target_mapping', 'cat_to_name.json')
    cat_to_name = get_cat_to_name_mapping(cat_to_name_file)

    if no_unique_train_labels != len(cat_to_name):
        print("WARN: Output layer size will be set to number of unique train dataset targets ({}). "
              "Output layer size does not match number of labels provided in {}".format(no_unique_train_labels,
                                                                                        cat_to_name_file))

    # Set size of output layer
    parameters['output_size'] = no_unique_train_labels

    # Create model based on parameters
    print('Create model ...')
    model = create_model(parameters)

    # Use negative log likelihood loss as error function (because of log softmax)
    criterion = nn.NLLLoss()

    # Use Adam optimizer for parameter optimization of classifier parameters after backpropagation
    arch = parameters.get('arch', 'vgg16')
    if arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr=parameters.get('learning_rate', 0.001))
    elif arch == 'vgg16' or arch == 'densenet161':
        optimizer = optim.Adam(model.classifier.parameters(), lr=parameters.get('learning_rate', 0.001))

    # Update parameters
    parameters['output_function'] = 'LogSoftmax'
    parameters['loss_function'] = 'NLLLoss'
    parameters['optimizer'] = 'Adam'

    # Define 'cuda' as device, if GPU is available and use of GPU confirmed via CLI argument
    print('Define device ...')
    device = get_device(parameters.get('gpu', False))

    # Set device for model (Use of GPU or CPU)
    model.to(device)

    # Define after which processed training batch, the train loss and accuracy will be printed
    # and the validation dataset be processed
    train_loss_accuracy_batch_interval = int(parameters.get('no_records_train') / parameters.get('batch_size') / 4)
    parameters['train_loss_accuracy_batch_interval'] = train_loss_accuracy_batch_interval

    # Print model
    print("Model:")
    print(model)

    # Print parameters
    print_parameters(parameters)

    # Lists for storing the loss and accuracies results
    train_losses, valid_losses, valid_accuracies = [], [], []

    # Train and validate image classifier
    print("Start training inclusive validation ...")
    train_image_classifier(parameters, device, model, criterion, optimizer, train_loader, valid_loader,
                           train_losses, valid_losses, valid_accuracies)

    # Finally, store train and validation results in dictionary
    parameters['train_losses'] = np.array(train_losses)
    parameters['valid_losses'] = np.array(valid_losses)
    parameters['valid_accuracies'] = np.array(valid_accuracies)

    # Lists for storing the loss and accuracies results
    test_losses, test_accuracies = [], []

    # Test the classifier
    print("Start testing the model ...")
    test_image_classifier(device, model, criterion, test_loader, test_losses, test_accuracies, calc_per_batch=True)

    # Finally, store test results in dictionary
    parameters['test_losses'] = np.array(test_losses)
    parameters['test_accuracies'] = np.array(test_accuracies)

    # Save parameters and model to checkpoint
    print('Save the model and parameters to checkpoint ...')
    save_checkpoint(model, parameters)

    # Define end time
    end_time_main = perf_counter()

    duration_in_s = end_time_main - start_time_main
    print("Elapsed time ({:d}s): {}".format(int(duration_in_s), get_day_hour_min_sec_str(duration_in_s)))


# Entry point for executing file via CLI
if __name__ == "__main__":
    main()
