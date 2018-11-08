import numpy as np
import scipy.misc
import shutil
import torch
from collections import OrderedDict
import scipy
import csv


def isfloat(x):
    """Checks to see if x can be a float

    Args:
        x: input as string (usually)

    Returns:
        bool: can or cannot be float

    """
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    """Checks to see if x can be an int

    Args:
        x: input as string (usually)

    Returns:
        bool: can or cannot be int

    """
    try:
        a = float(x)
        b = int(a) if a != float('Inf') else 9999999
    except ValueError:
        return False
    else:
        return a == b


def save_statistics(log_dir, statistics_file_name, list_of_statistics, create=False):
    """Saves training stats as csv file for use later

    Args:
        log_dir: Where to save the file
        statistics_file_name: Filename of CSV
        list_of_statistics: To append to the last line of the CSV
        create: True if making new CSV otherwise False


    """
    if create:
        with open("{}/{}.csv".format(log_dir, statistics_file_name), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(list_of_statistics)
    else:
        with open("{}/{}.csv".format(log_dir, statistics_file_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(list_of_statistics)


def load_statistics(log_dir, statistics_file_name):
    """Loads CSV train stats file into dictionary

    Args:
        log_dir: directory of CSV stats file
        statistics_file_name: filename of CSV

    Returns:
        A dictionary with training stats accrued so far

    """
    data_dict = dict()
    with open("{}/{}.csv".format(log_dir, statistics_file_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n", "").replace("\r", "").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n", "").replace("\r", "").split(",")
            for key, item in zip(data_labels, data):
                if item not in data_labels:
                    data_dict[key].append(int(float(item)) if isint(item) else float(item))
    return data_dict


def save_image_batch(filename, images):
    """Saves a batch of images to view

    Can be used to save input images/outputs of an autoencoder/features/etc

    If using to save features, these must be in the same layout as above.

    Args:
        filename: where to save output image
        images: input batch of images as Tensors of: [B, C, H, W], where
            B - Batch size
            C - Channels - either 1 or 3 (if you want to save features, reshape these first)
            W - Width (spatial)
            H - Height (spatial)
    """

    n = images.size(0)
    width = int(np.round(np.sqrt(n)))
    height = int(np.ceil(n/width))

    img_w = images.size(2)
    img_h = images.size(3)
    img_c = images.size(1)
    buffer = 1

    output_img = np.zeros(shape=(buffer + img_w * width + buffer * width, buffer + img_h * height + buffer * height, img_c)) + 1
    for i, img in enumerate(images):
        x, y = np.unravel_index(i, dims=(width, height))
        img = np.transpose(img.detach().cpu().numpy(), (1,2,0))
        img[:,:,0] = img[:,:,0] * 0.5 + 0.5
        img[:,:,1] = img[:,:,1] * 0.5 + 0.5
        img[:,:,2] = img[:,:,2] * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        output_img[buffer + x * (img_w + buffer): buffer + (x) * (img_w + buffer) + img_w, buffer + y * (img_h + buffer): buffer + y * (img_h + buffer) + img_h, :] = img

    scipy.misc.toimage(np.squeeze(output_img)).save(filename)



def save_checkpoint(state, is_best, directory='', filename='checkpoint.pth.tar'):
    """Saves pytorch model state

    Args:
        state: with network and optimizer
        is_best: gives a different filename for the best model
        directory: where to save the checkpoint
        filename: filename of checkpoint
    """
    save_path = '{}/{}'.format(directory, filename) if directory != '' else filename
    torch.save(state, save_path)

    if is_best:
        best_save_path = '{}/best_{}'.format(directory, filename) if directory != '' else 'best_{}'.format(filename)
        shutil.copyfile(save_path, best_save_path)


def restore_model(net, optimizer_c, optimizer_g, args):
    """Restores model and optimizer from file

    Args:
        net: network to restore
        optimizer: optimizer to restore
        args: contains filepaths for restoration

    Returns:

    """
    if args.resume:
        restore_path = '{}'.format(args.latest_loadpath)
        print('Latest, continuing from {}'.format(restore_path))
        checkpoint = torch.load(restore_path, map_location=lambda storage, loc: storage)

        new_state_dict = OrderedDict()
        for k, v in checkpoint['net'].items():
            if 'module' in k and args.device == 'cpu':
                name = k.replace("module.", "")  # remove module.
            else:
                name = k
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        for k, v in checkpoint['optimizer_c'].items():
            if 'cuda' in k and args.device == 'cpu':
                name = k.replace("cuda..", "")  # remove module.
            else:
                name = k
            new_state_dict[name] = v
        optimizer_c.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        for k, v in checkpoint['optimizer_g'].items():
            if 'cuda' in k and args.device == 'cpu':
                name = k.replace("cuda..", "")  # remove module.
            else:
                name = k
            new_state_dict[name] = v
        optimizer_g.load_state_dict(new_state_dict)


def build_experiment_folder(args):
    """Builds experiment folder

    Args:
        args: contains experiment name and logs filepath

    Returns:

    """
    experiment_name, log_path = args.exp_name, args.logs_path
    saved_models_filepath = "{}/{}/{}".format(log_path, experiment_name.replace("%.%", "/"), "saved_models")
    logs_filepath = "{}/{}/{}".format(log_path, experiment_name.replace("%.%", "/"), "summary_logs")
    images_filepath = "{}/{}/{}".format(log_path, experiment_name.replace("%.%", "/"), "images")

    import os

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)
    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)
    if not os.path.exists(images_filepath):
        os.makedirs(images_filepath)

    if not os.path.exists(images_filepath + '/train'):
        os.makedirs(images_filepath + '/train')

    if not os.path.exists(images_filepath + '/test'):
        os.makedirs(images_filepath + '/test')

    args.saved_models_filepath = saved_models_filepath
    args.logs_filepath = logs_filepath
    args.images_filepath = images_filepath
    return saved_models_filepath, logs_filepath, images_filepath

