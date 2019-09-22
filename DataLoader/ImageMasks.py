"""
Handle RLE mask data. This can load from either a single class or multiple classes either in a given row
or with each class in its own row.

"""
from os import getcwd, mkdir
from pathlib import Path

import pandas as pd
from fastai.vision import *
import tqdm


def _setup_paths(input_path, output_path):
    '''

    Parameters
    ----------
    input_path
    output_path

    Returns
    -------

    '''
    if output_path is None:
        output_path = Path(getcwd(), 'masks')
        mkdir(output_path)
    else:
        output_path = Path(output_path)

    if input_path is None:
        input_path = Path(getcwd())
    else:
        input_path = Path(input_path)
    return input_path, output_path


def _image_shape(img_path):
    img = open_image(img_path)
    return img.shape


def masks_from_df(df: pd.DataFrame, filename_col: str, mask_col: str, output_path=None, input_path=None) -> None:
    """

    Parameters
    ----------
    df              A pandas dataframe containing the image filename names and mask data
    filename_col    Which column in the dataframe has the filename
    mask_col        Which column in the dataframe has the mask data
    output_path     A path to output mask data data to
    input_path      A path to the input images

    Returns
    -------

    """
    input_path, output_path = _setup_paths(input_path, output_path)
    for row in tqdm(df.iterrows()):
        shape = _image_shape()


def multilabel_masks_from_df(df: pd.DataFrame, filename_col: str, mask_col: str, labels: list, output_path=None,
                             input_path=None) -> None:
    """

    Parameters
    ----------
    df
    filename_col
    mask_col
    labels
    output_path
    input_path

    Returns
    -------

    """
    input_path, output_path = _setup_paths(input_path, output_path)
    mask_value = {label: i+1 for i, label in enumerate(labels)}
