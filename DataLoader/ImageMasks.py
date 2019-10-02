"""
Handle RLE mask data. This can load from either a single class or multiple classes either in a given row
or with each class in its own row.

"""
from os import getcwd, mkdir
from pathlib import Path
import re

import pandas as pd
import numpy as np
from fastai.vision import *
from tqdm import tqdm


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
    for _, row in tqdm(df.iterrows()):
        shape = _image_shape(Path(input_path, row[filename_col]))
        decoded = rle_decode(row[mask_col], shape[1:])
        file_output_path = output_path.joinpath(row[filename_col]).with_suffix('.npz')
        np.save(file_output_path, decoded)


def multilabel_masks_from_df(df: pd.DataFrame, filename_col: str, labels: list, output_path=None,
                             input_path=None) -> None:
    """

    Parameters
    ----------
    df
    filename_col
    labels
    output_path
    input_path

    Returns
    -------

    """
    input_path, output_path = _setup_paths(input_path, output_path)
    mask_value = {label: i+1 for i, label in enumerate(labels)}
    for _, row in tqdm(df.iterrows()):
        shape = _image_shape(Path(input_path, row[filename_col]))
        mask = np.zeros(shape[1:])
        for label in labels:
            if row[label] is np.nan:
                continue
            decoded = rle_decode(row[label], shape[1:]) * mask_value[label]
            # TODO there should be a check to make sure each pixel is assigned to only 1 class, and
            # if not set it to 0, since it is likely not that informative
            mask += decoded
        file_output_path = output_path.joinpath(row[filename_col]).with_suffix('.npz')
        np.save(file_output_path, mask)


def prepare_multilabel_df(df: pd.DataFrame, input_col: str, rle_col: str) -> pd.DataFrame:
    """
    This utility processes a dataframe that has a different row for each label and moves
    the encodings into their own columns instead.

    Parameters
    ----------
    df

    Returns
    -------

    """
    temp_df = deepcopy(df)
    temp_df['class'] = temp_df[input_col].str.extract(r'(.+)_(.*)')[1]
    temp_df['real_filename'] = df[input_col].str.extract('([^_]+)')
    image_classes = set(temp_df['class'])
    rows = []
    for name, group in tqdm(temp_df.groupby('real_filename')):
        row = {'filename': name}
        for idx, group_row in group.iterrows():
            row[group_row['class']] = group_row[rle_col]
        rows.append(row)
    new_df = pd.DataFrame.from_dict(rows)
    return new_df, image_classes
