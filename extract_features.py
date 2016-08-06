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

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import scipy.misc
import skimage.color
import tqdm

import vggnet


def crop_center(im):
    """
    Crops the center out of an image.

    Args:
        im (numpy.ndarray): Input image to crop.
    Returns:
        numpy.ndarray, the cropped image.
    """

    h, w = im.shape[0], im.shape[1]

    if h < w:
        return im[0:h,int((w-h)/2):int((w-h)/2)+h,:]
    else:
        return im[int((h-w)/2):int((h-w)/2)+w,0:w,:]


def preprocess_image(im):
    """
    Preprocesses an image for a VGG model.

    Args:
        im (numpy.ndarray): Input RGB image to preprocess.
    Returns:
        numpy.ndarray, the preprocessed image.
    """

    # Resize if needed
    if im.shape[0:1] != vggnet.IMAGE_SIZE:
        im = scipy.misc.imresize(im, vggnet.IMAGE_SIZE)

    # Mean-center image
    out = np.empty((3,)+vggnet.IMAGE_SIZE)
    out[2,:,:] = im[:,:,0]-vggnet.R_MEAN
    out[1,:,:] = im[:,:,1]-vggnet.G_MEAN
    out[0,:,:] = im[:,:,2]-vggnet.B_MEAN

    return out


def process_directory(input_dir, output_dir, model, compute_motion=False, batch_size=32):
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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif os.path.isfile(output_dir):
        raise RuntimeError("Output directory '{}' is a file!".format(output_dir))

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

        filename = os.path.split(filepath)[-1]

        # Estimate number of frames
        num_frames = int(clip.fps*clip.duration)

        visual_frames = np.empty((batch_size,3)+vggnet.IMAGE_SIZE)
        motion_frames = np.empty((batch_size,3)+vggnet.IMAGE_SIZE)

        visual_features = np.empty((num_frames, 4096))
        motion_features = np.empty((num_frames, 4096))

        message = "\rProcessing {name} ({{:{frames}d}}/{{:{frames}d}})".format(
                name=filename, frames=len(str(num_frames)))

        frame_idx = 0
        feat_idx = 0
        actual_frames = 0

        # Temp matrix to hold motion
        motion = np.zeros((3,)+vggnet.IMAGE_SIZE)

        sys.stdout.write(message.format(0, num_frames))
        sys.stdout.flush()

        for frame in clip.iter_frames():

            cropped = scipy.misc.imresize(crop_center(frame), vggnet.IMAGE_SIZE)

            motion[2,:,:] = motion[1,:,:]
            motion[1,:,:] = motion[0,:,:]
            motion[0,:,:] = 255*skimage.color.rgb2gray(cropped)

            motion_frames[frame_idx,:,:,:] = preprocess_image(motion)
            visual_frames[frame_idx,:,:,:] = preprocess_image(cropped)

            frame_idx += 1
            actual_frames += 1

            if frame_idx == batch_size:
                vis_feat = model.predict( visual_frames, batch_size=batch_size )
                mot_feat = model.predict( motion_frames, batch_size=batch_size )

                visual_features[feat_idx:feat_idx+batch_size,:] = vis_feat
                motion_features[feat_idx:feat_idx+batch_size,:] = mot_feat

                frame_idx = 0
                feat_idx += batch_size

                sys.stdout.write(message.format(feat_idx, num_frames))
                sys.stdout.flush()

        sys.stdout.write(message.format(num_frames, num_frames))
        sys.stdout.write('\n')
        sys.stdout.flush()

        np.save( os.path.join(output_dir, filename+'.visual'), visual_features )
        np.save( os.path.join(output_dir, filename+'.motion'), motion_features )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description = ""
    )

    parser.add_argument('data',
            help="Directory of videos to process.")
    parser.add_argument('-o', '--output', default='output',
            help="Directory where extracted features should be stored.")

    parser.add_argument('-b', '--batch_size', default=32, type=int,
            help="Number of frames to be processed each batch.")
    parser.add_argument('--weights', default='vgg19_weights.h5',
            help="Pretrained weights file for VGG-19. If file doesn't exist, "
                 "the weights will be downloaded instead.")

    args = parser.parse_args()

    model = vggnet.VGG_19(args.weights)

    print()
    process_directory(args.data, args.output, model, batch_size=args.batch_size)
    print("\nFinished!\n")

