# import the necessary packages
from tensorflow.io import FixedLenFeature
from tensorflow.io import parse_single_example
from tensorflow.io import parse_tensor
from tensorflow.image import flip_left_right
from tensorflow.image import rot90
import tensorflow as tf
# define AUTOTUNE object
AUTO = tf.data.AUTOTUNE
def random_crop(lrImage, hrImage, hrCropSize=96, scale=4):
    # calculate the low resolution image crop size and image shape
    lrCropSize = hrCropSize // scale
    lrImageShape = tf.shape(lrImage)[:2]
    # calculate the low resolution image width and height offsets
    lrW = tf.random.uniform(shape=(),
        maxval=lrImageShape[1] - lrCropSize + 1, dtype=tf.int32)
    lrH = tf.random.uniform(shape=(),
        maxval=lrImageShape[0] - lrCropSize + 1, dtype=tf.int32)
    # calculate the high resolution image width and height
    hrW = lrW * scale
    hrH = lrH * scale
    # crop the low and high resolution images
    lrImageCropped = tf.slice(lrImage, [lrH, lrW, 0], 
        [(lrCropSize), (lrCropSize), 3])
    hrImageCropped = tf.slice(hrImage, [hrH, hrW, 0],
        [(hrCropSize), (hrCropSize), 3])
    # return the cropped low and high resolution images
    return (lrImageCropped, hrImageCropped)
   
def get_center_crop(lrImage, hrImage, hrCropSize=96, scale=4):
    # calculate the low resolution image crop size and image shape
    lrCropSize = hrCropSize // scale
    lrImageShape = tf.shape(lrImage)[:2]
    # calculate the low resolution image width and height
    lrW = lrImageShape[1] // 2
    lrH = lrImageShape[0] // 2
    # calculate the high resolution image width and height
    hrW = lrW * scale
    hrH = lrH * scale
    # crop the low and high resolution images
    lrImageCropped = tf.slice(lrImage, [lrH - (lrCropSize // 2),
        lrW - (lrCropSize // 2), 0], [lrCropSize, lrCropSize, 3])
    hrImageCropped = tf.slice(hrImage, [hrH - (hrCropSize // 2),
        hrW - (hrCropSize // 2), 0], [hrCropSize, hrCropSize, 3])
    # return the cropped low and high resolution images
    return (lrImageCropped, hrImageCropped)
    
def random_flip(lrImage, hrImage):
    # calculate a random chance for flip
    flipProb = tf.random.uniform(shape=(), maxval=1)
    (lrImage, hrImage) = tf.cond(flipProb < 0.5,
        lambda: (lrImage, hrImage),
        lambda: (flip_left_right(lrImage), flip_left_right(hrImage)))
    
    # return the randomly flipped low and high resolution images
    return (lrImage, hrImage)

def random_rotate(lrImage, hrImage):
    # randomly generate the number of 90 degree rotations
    n = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # rotate the low and high resolution images
    lrImage = rot90(lrImage, n)
    hrImage = rot90(hrImage, n)
    # return the randomly rotated images
    return (lrImage, hrImage)

def read_train_example(example):
    # get the feature template and  parse a single image according to
    # the feature template
    feature = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string),
    }
    example = parse_single_example(example, feature)
    # parse the low and high resolution images
    lrImage = parse_tensor(example["lr"], out_type=tf.uint8)
    hrImage = parse_tensor(example["hr"], out_type=tf.uint8)
    # perform data augmentation
    (lrImage, hrImage) = random_crop(lrImage, hrImage)
    (lrImage, hrImage) = random_flip(lrImage, hrImage)
    (lrImage, hrImage) = random_rotate(lrImage, hrImage)
    # reshape the low and high resolution images
    lrImage = tf.reshape(lrImage, (24, 24, 3))
    hrImage = tf.reshape(hrImage, (96, 96, 3))
    # return the low and high resolution images
    return (lrImage, hrImage)

def read_test_example(example):
    # get the feature template and  parse a single image according to
    # the feature template
    feature = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string),
    }
    example = parse_single_example(example, feature)
    # parse the low and high resolution images
    lrImage = parse_tensor(example["lr"], out_type=tf.uint8)
    hrImage = parse_tensor(example["hr"], out_type=tf.uint8)
    # center crop both low and high resolution image
    (lrImage, hrImage) = get_center_crop(lrImage, hrImage)
    # reshape the low and high resolution images
    lrImage = tf.reshape(lrImage, (24, 24, 3))
    hrImage = tf.reshape(hrImage, (96, 96, 3))
    # return the low and high resolution images
    return (lrImage, hrImage)

def load_dataset(filenames, batchSize, train=False):
    # get the TFRecords from the filenames
    dataset = tf.data.TFRecordDataset(filenames, 
        num_parallel_reads=AUTO)
    # check if this is the training dataset
    if train:
        # read the training examples
        dataset = dataset.map(read_train_example,
            num_parallel_calls=AUTO)
    # otherwise, we are working with the test dataset
    else:
        # read the test examples
        dataset = dataset.map(read_test_example,
            num_parallel_calls=AUTO)
    # batch and prefetch the data
    dataset = (dataset
        .shuffle(batchSize)
        .batch(batchSize)
        .repeat()
        .prefetch(AUTO)
    )
    # return the dataset
    return dataset
