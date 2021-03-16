from time import perf_counter
from PIL import Image
import train
from pathlib import Path
import pathlib
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
import os


# global variables
supported_file_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp')


def get_args():
    """
    Validate and return parsed CLI arguments as dictionary

    :return:
    """
    # Default root directory will be the directory in which this file is located
    default_root_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='Predict image classification based on existing pretrained model')
    parser.add_argument('checkpoint', type=lambda p: train.path_exists(p, replace_keyword="checkpoint=", is_file=True),
                        metavar='<path to checkpoint file>', nargs='?',
                        help='file path to torch checkpoint of the existing pretrained model')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--image_file', type=lambda p: train.path_exists(p, is_file=True),
                       metavar='<path to image file>', nargs='?', help='use path to image file')
    group.add_argument('-i', '--image_dir', type=lambda p: train.path_exists(p), metavar='<path to images dir>',
                       nargs='?', help='use directory containing image files; can be used as alternative to specifying'
                                       ' just a single image file path')
    group.add_argument('-t', '--test_mode', action='store_true', help='use test image dataset instead of '
                                                                      'image file or directory; defaults to False')
    parser.add_argument('-d', '--test_data_dir', type=lambda p: train.path_exists(p),
                        metavar='<path to test data dir>', nargs='?',
                        default=default_root_dir.joinpath('flowers', 'test'),
                        help='Will be used only when test_mode is enabled. Directory containing the test data '
                             'as expected by torchvision.datasets.ImageFolder; defaults to <script_dir>/flowers/test')
    parser.add_argument('-k', '--top_k', type=int, metavar='<integer>', nargs='?', default=5,
                        help='number of top k most likely classes to be returned')
    parser.add_argument('-g', '--gpu', action='store_true', help='use gpu, if available; defaults to False')
    parser.add_argument('-c', '--category_names', type=lambda p: train.path_exists(p, is_file=True),
                        metavar='<path to json file>', nargs='?', default=default_root_dir.joinpath('cat_to_name.json'),
                        help='JSON file containing a mapping of categories and label names for targets'
                             'defaults to cat_to_name.json')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='increase output verbosity; defaults to 0')

    try:
        # vars returns dictionary
        return vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))


def load_model_from_checkpoint(file, device):
    """
    Load and initialize existing pretrained model from torch checkpoint file

    :param file: Path to torch checkpoint file
    :param device: torch.device
    :return model: Pretrained model
    :return parameters: Dictionary containing model parameters and training results saved with checkpoint
    """

    if device == 'cuda':
        # Load all tensors onto GPU
        map_location = lambda storage, loc: storage.cuda()
    else:
        # Load all tensors onto CPU
        map_location = lambda storage, loc: storage

    # Assuming model was trained and checkpoint saved on Linux, but predict.py inference is executed using Windows.
    # Then, it is required to implement the following quick fix, because otherwise the exception is raised:
    # "NotImplementedError: cannot instantiate 'PosixPath' on your system"
    # Credits to https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath
    if type(file) == pathlib.WindowsPath:
        tmp_PosixPath = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath

    parameters = torch.load(file, map_location=map_location)

    # Restore default
    if type(file) == pathlib.WindowsPath:
        pathlib.WindowsPath = pathlib.PosixPath
        pathlib.PosixPath = tmp_PosixPath

    model = train.create_model(parameters)

    model.class_to_idx = parameters.get('train_datasets_class_to_idx')
    model.load_state_dict(parameters.get('state_dict'), strict=False)

    return model, parameters


def get_class_label_names(class_idx_predictions, class_to_idx, class_to_label):
    """
    Returns the mapped label names of the classification index predictions as list.
    In case the index does not exist, classification index predicted is returned as label.

    :param class_idx_predictions: torch.Tensor, top classification index values predicted by the model
    :param class_to_idx: Dictionary, mapping between model classification index and
    numbered classification labels determined by data loader ImageFolder (see directory structure)
    :param class_to_label: Dictionary, mapping between numbered classification labels and label names
    :return class_labels_predictions: list, mapped label names of the classifications index prediction input
    """
    # Flatten to 1D tensor and convert to ndarray
    class_idx_predictions = np.array(np.squeeze(class_idx_predictions))

    # Switch key to value and value to key
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    # class_idx_predictions represents an class index, e.g. provided by model prediction
    # Get the label from the class that matches the index
    class_labels_predictions = [class_to_label.get(idx_to_class.get(idx, None), idx) for idx in class_idx_predictions]

    # Return list
    return class_labels_predictions


def get_torch_dataset_images_labels(data_dir):
    """
    Returns the images and labels loaded by torchvision.datasets.ImageFolder from data_dir directory

    :param data_dir: Directory containing the data as expected by torchvision.datasets.ImageFolder
    :return test_datasets.imgs: List containing for each image an tuple consisting of Path (str) and Label (Any)
    """

    # Transforms for testing sets
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])

    # Definition of testing datasets using ImageFolder
    test_datasets = datasets.ImageFolder(data_dir, transform=test_transforms)

    return test_datasets.imgs


def get_images(args):
    """
    Get the image path and labels depending on provided CLI arguments:
    - Single image file
    - Directory tree containing many images
    - Data directory structured as expected by torchvision.datasets.ImageFolder

    :param args: CLI arguments provided
    :return: List that contains tuples which consists of image path (str) and label (Any)
    """

    print("Get image path and labels ...")

    # Initialize variables
    image_dir = args.get('image_dir', None)
    test_mode = args.get('test_mode', False)
    v = args.get('verbosity', 0)

    dataset_images_labels = []

    # Load image from file path, if image_dir and test_mode is not specified
    if image_dir is None and not test_mode:
        print("... for a single image file")
        # Add path to file and value -1 as label to dataset list
        dataset_images_labels = [(args.get('image_file'), -1)]

    # Get image files from image dir path
    elif image_dir is not None:
        print("... for all images found in directory tree with top: {}".format(image_dir))

        if v >= 1:
            print("Supported image file extensions: ", supported_file_extensions)

        # Walk top down to directory tree with top image_dir
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                # Check, if file ends with an supported file extension
                if f.lower().endswith(supported_file_extensions):
                    # Add path to file and value -1 as label to dataset list
                    dataset_images_labels.append((os.path.join(root, f), -1))

    # Get images from test dataset
    elif test_mode:
        # Get test data directory
        test_data_dir = str(args.get('test_data_dir'))

        # In case the default parameter for option test_data_dir in argument parser was used,
        # it's not clear if the directory exists. Check if directory exists and raise an exception
        # in case the dir path does not exist.
        train.path_exists(test_data_dir)
        dataset_images_labels = get_torch_dataset_images_labels(test_data_dir)

    return dataset_images_labels


def process_image(pil_im):
    """
    Process PIL image for use in a PyTorch model.
    Image will be scaled, cropped, and normalized and converted to a numpy array.

    :param pil_im:
    :return ndarray:
    """
    # Target sizes
    new_min_max_width_or_height = 256
    crop_len = 224

    # Current image size
    width, height = pil_im.size

    # Calculate the ratio based on new width and current width
    ratio = new_min_max_width_or_height / width

    # Calculate the new height based on new_width/width ratio
    new_height = int(height * ratio)

    # If new_height is less than expected new height, then calculate ratio based new height instead of new width
    if new_height < new_min_max_width_or_height:
        ratio = new_min_max_width_or_height / height
        new_width = int(width * ratio)
        new_height = new_min_max_width_or_height
    else:
        new_width = new_min_max_width_or_height

    # Resize the image to new_width and new_height
    new_size = (new_width, new_height)
    pil_im = pil_im.resize(new_size)

    # Input dimensions when plotting pil_image will look like:
    # Left, Top = 0, 0
    # Right, Bottom = 256 or greater, 256 or greater

    # Crop the center 224x224 portion of the image
    left = (pil_im.width - crop_len) / 2
    top = (pil_im.height - crop_len) / 2
    right = left + crop_len
    bottom = top + crop_len

    pil_im = pil_im.crop((left, top, right, bottom))

    # Convert from PIL image to numpy array and change channel encoding from 0-255 to floats from 0-1
    # np_im = np.array(pil_im)
    np_im = np.array(pil_im) / 255

    # Color channel normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_im = (np_im - mean) / std

    # The color channel is the third dimension in the PIL image and Numpy array.
    # It needs to be the first dimension. The order of the other two dimensions need to be retained.
    return np_im.transpose((2, 0, 1))


def predict(image_path, model, device, topk=5):
    """
    Predict the classes of an image using a trained deep learning model and return the
    most likely topk predicted classes and their probabilities

    :param device: torch.device indicating if GPU or CPU will be used
    :param image_path: path to the image file provided as string OR
    Image already transformed (resized, cropped, color channel normalized) provided as torch.Tensor
    :param model: model used for prediction
    :param topk: number of top prediction probabilities and classes to be return
    :return top_p: top prediction probabilities
    :return top_k: top prediction classes
    """

    # Image path is path to an image file
    if type(image_path) in (str, pathlib.WindowsPath, pathlib.PosixPath):
        # Open image as PIL image
        try:
            with Image.open(image_path) as im:
                # Preprocessed the image for pytorch model
                im = process_image(im)
        except OSError as e:
            print("ERROR: Skip the image, because it failed to opened. Error: {}".format(e))
            return None, None
        # Convert to ndarray to tensor and respective type depending on, if GPU is available
        im_t = torch.tensor(im, dtype=torch.double, device=device)

        # Ensure that image tensor is converted to FloatTensor type
        im_t = im_t.type(torch.FloatTensor)

    # image_path already is a tensor
    elif torch.is_tensor(image_path):

        im_t = image_path

    # Add the batch dimension to the tensor (train )
    im_t = torch.unsqueeze(im_t, 0)

    # Enable test mode, which disables dropouts
    model.eval()

    with torch.no_grad():
        # Forward feed the model with the image
        output = model.forward(im_t)

        # Calculate probability (because model returns logsoftmax)
        ps = torch.exp(output)

        # Get the best probability value and it's index
        top_p, top_k = ps.topk(topk, dim=1)

    return top_p, top_k


def main():
    """
    Main procedure

    :return:
    """

    # Define start time
    start_time_main = perf_counter()

    # Get CLI arguments
    print('Get CLI arguments ...')
    args = get_args()
    v = args.get('verbosity', 0)

    if v >= 1:
        print('CLI arguments:')
        print(args)

    # Check if checkpoint does no exist, stop
    checkpoint_file = args.get('checkpoint', None)
    if checkpoint_file is None:
        print('ERROR: Path to torch checkpoint file is not defined, see help: {} -h'.format(__file__))
        exit()
    if v >= 1:
        print('Checkpoint file: ', checkpoint_file)

    # Define 'cuda' as device, if GPU is available and use of GPU confirmed via CLI argument
    print('Define device ...')
    device = train.get_device(args.get('gpu', False))

    if v >= 1:
        print('Device: ', device)

    # Load the model from checkpoint
    print('Load the model and parameters from checkpoint ...')
    model, parameters = load_model_from_checkpoint(checkpoint_file, device)

    if v >= 2:
        print('Parameters: ')
        print(parameters)
        print('Model:')
        print(model)
    elif v >= 1:
        train.print_parameters(parameters)
        print('ModeL:')
        print(model)

    # Load data
    dataset_image_labels = get_images(args)

    # Get label names of the classifications as dictionary
    print('Load category names ...')
    cat_to_name = train.get_cat_to_name_mapping(args.get('category_names'))

    if v >= 1:
        print('Mapping of categories to names: {}'.format(args.get('category_names')))
        print(cat_to_name)

    # Number of most likely classifications per prediction
    k = args.get('top_k', 5)
    if k == 0:
        print("WARN: --top_k was set to 0 and will be reset to default value 5.")
        k = 5

    print("Let's predict ...")
    # For each image predict the classification
    for i in range(len(dataset_image_labels)):

        # Get image path and label
        image_path, label = dataset_image_labels[i]

        # Get probability and class
        top_p, top_k = predict(image_path, model, device, topk=k)

        # Error occurred, because false, false was returned
        if top_p is None and top_k is None:
            continue

        # Dictionary cat_to_name will be empty in case no category to name mapping file exists
        if cat_to_name:
            # Get category names of top k predictions
            # Just the top 1 predication was requested
            if k == 1:
                categories = [cat_to_name.get(cls, "Unknown") for cls, idx in model.class_to_idx.items() if top_k == idx]
                if categories:
                    categories = categories[0]
                else:
                    categories = "Unknown"
            # Multiple top predication were requested
            else:
                categories = get_class_label_names(top_k, model.class_to_idx, cat_to_name)

            # Get expected category name for label number
            expected_category = [cat_to_name.get(cls, "Unknown") for cls, idx in model.class_to_idx.items() if label == idx]
            if expected_category:
                expected_category = expected_category[0]
            else:
                expected_category = "Unknown"
        else:
            # Use class idx as categories, flatten to 1D tensor
            categories = np.array(np.squeeze(top_k))
            expected_category = label if label != -1 else "Unknown"

        print("\nImage #{}: {}"
              "\nExpected category: {}"
              "\nTop {} prediction(s):".format(i+1, image_path, expected_category, k))

        if k == 1:
            print("{:<10.5%} {}".format(np.squeeze(top_p), categories))
        else:
            # Flatten to 1D tensor and calculate percentage
            ps_sci = np.array(np.squeeze(top_p))

            for c, ps in zip(categories, ps_sci):
                print("{:<10.5%} {}".format(ps, c))

    # Define end time
    end_time_main = perf_counter()

    duration_in_s = end_time_main - start_time_main
    print("Elapsed time ({:d}s): {}".format(int(duration_in_s), train.get_day_hour_min_sec_str(duration_in_s)))


# Entry point for executing file via CLI
if __name__ == "__main__":
    main()
