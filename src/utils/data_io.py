"""
This file defines IO utils for the project
"""

import os
import pathlib

import numpy as np

from torch import Tensor
from skimage.io import imread, imshow
from matplotlib import pyplot as plt


def load_image(file_name: str, data_dir: str = "faces") -> np.ndarray:
    """
    Loads the image located at ../data/data_dir/file_name
    :param file_name: The name of the file to load
    :param data_dir: The directory in the data parent to load from
    :return: The image as a numpy array
    """
    if not file_name.startswith("/"):
        file_name = f"{get_data_path()}/{data_dir}/{file_name}"

    return imread(file_name)


def load_keypoints(image_path: str, data_dir: str = "faces") -> np.ndarray:
    """
    Loads the keypoints for a given image
    :param image_path: The path of the image to get keypoints from
    :param data_dir: The subdir of data to load from
    :return: The list of keypoints
    """
    # Load the image to get height and width
    image = load_image(image_path, data_dir)
    height, width = image.shape[:2]

    # Full path to image
    if not image_path.startswith("/"):
        image_path = f"{get_data_path()}/{data_dir}/{image_path}"

    # The path to the keypoint file
    keypoint_file = os.path.splitext(image_path)[0] + ".asf"
    keypoints = np.genfromtxt(keypoint_file, skip_header=16, skip_footer=1)[:, 2:4]

    # Scale the keypoints correctly
    keypoints = (keypoints @ np.array([[width, 0], [0, height]])).astype(np.int)

    return keypoints


def show_image(image) -> None:
    """
    Shows an image
    :param image: The image to show
    :return: None
    """

    if isinstance(image, Tensor):
        image = image.squeeze()
        image = image.numpy()

        if len(image.shape) > 2:
            image = image.transpose((1, 2, 0))

    keyword_args = {}

    if len(image.shape) < 3 or image.shape[2] == 1:
        keyword_args["cmap"] = "gray"

    imshow(image, **keyword_args)
    plt.show()


def show_image_with_keypoints(image: np.ndarray, keypoints: np.ndarray) -> None:
    """
    Shows an image plotted with the given keypoints
    :param image: The image to show
    :param keypoints: The keypoints to plot
    :return: None
    """
    # Reshape from tensor if necessary
    if isinstance(image, Tensor):
        image = image.squeeze()
        image = image.numpy()

        if len(image.shape) > 2 and image.shape[2] != 3:
            image = image.transpose((1, 2, 0))

    # If single keypoint then expand dims
    keypoints = keypoints.squeeze()
    if len(keypoints.shape) == 1:
        keypoints = keypoints.reshape((1, 2))

    keyword_args = {}

    if len(image.shape) < 3 or image.shape[2] == 1:
        keyword_args["cmap"] = "gray"

    plt.imshow(image, **keyword_args)
    plt.scatter(keypoints[:, 0], keypoints[:, 1])
    plt.show()


def get_src_path() -> str:
    """
    Gets the path to the src/ directory in absolute terms
    :return: The relevant path
    """
    return f"{pathlib.Path(__file__).parent.parent.absolute()}"


def get_data_path() -> str:
    """
    Gets the absolute path to the data directory for this project
    :return: The path to the data
    """
    return f"{pathlib.Path(__file__).parent.parent.absolute()}/data"


def get_model_path() -> str:
    """
    Gets the absolute path to the saved directory for this project
    :return: the path to the saved models
    """
    return f"{pathlib.Path(__file__).parent.parent.absolute()}/saved"
