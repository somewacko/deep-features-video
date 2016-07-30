#!/usr/bin/env python3
"""
extract_features.py

Script to extract CNN features from video frames.

"""

from __future__ import print_function

import argparse
import os
import sys
import urllib

from moviepy.editor import VideoFileClip
import numpy as np

import vggnet


def preprocess_image(im):
    """
    Preprocesses an image for a VGG model by:

        1. Cropping out a center square of the frame
        2. Resizing to 224x224
        3. Mean-centering the image

    Args:
        im (numpy.ndarray): Input RGB image to preprocess.
    Returns:
        numpy.ndarray, the preprocessed image.
    """

    # TODO: Sanity check that preprocessing is correct

    # Crop out center of frame
    h, w = im.shape[0], im.shape[1]
    if h < w:
        im = frame[0:h,int((w-h)/2):int((w-h/2))+h,:]
    else:
        im = frame[int((h-w)/2):int((h-w/2))+w,0:w,:]

    assert(im.shape[0] == im.shape[1])

    # Resize to 224
    im = scipy.misc.imresize(im, (224, 224))

    out = np.empty((3,)+vggnet.IMAGE_SIZE)
    out[0,:,:] = im[:,:,2]-vggnet.B_MEAN
    out[1,:,:] = im[:,:,1]-vggnet.G_MEAN
    out[2,:,:] = im[:,:,0]-vggnet.R_MEAN

    return out


def process_directory(input_dir, output_dir, model, compute_motion=False):
    """
    Extracts CNN features from all videos in a directory.

    Args:
        input_dir (str): Input directory to extract from.
        output_dir (str): Directory where features should be stored.
        model (keras.model): VGG model to extract features with.
        compute_motion (bool): Whether or not motion features should be
            extracted as well.
    """

    # Validate args

    if not os.path.isdir(input_dir):
        raise RuntimeError("Input directory '{}' not found!".format(input_dir))

    if os.path.exists(output_dir):
        # TODO: Allow overwrite
        raise RuntimeError("Output directory '{}' already exists!".format(output_dir))

    os.makedirs(output_dir)

    filepaths = [
        os.path.join(input_dir, x)
        for x in os.listdir(input_dir)
            if os.path.isfile( os.path.join(input_dir, x) )
    ]

    for filepath in filepaths:

        try:
            clip = VideoFileClip(filepath)
        except Exception as e:
            print("Unable to read {}. Skipping...")
            print("Exception:", e)
            continue

        # Estimate number of frames
        num_frames = int(clip.fps*clip.duration)

        features = np.empty((num_frames, 4096))

        for idx, frame in enumerate(clip.iter_frames()):
            features[idx,:] = model.predict( preprocess_image(frame) )

        # TODO: Save features to file


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description = ""
    )

    parser.add_argument('data',
            help="Directory of videos to process.")
    parser.add_argument('-o', '--output', default='output',
            help="Directory where extracted features should be stored.")

    parser.add_argument('--motion', action='store_true',
            help="Also extract motion features as well.")
    parser.add_argument('--weights', default='vgg19.weights.h5',
            help="Pretrained weights file for VGG-19. If file doesn't exist, "
                 "the weights will be downloaded instead.")

    args = parser.parse_args()


