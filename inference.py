# USAGE
# python inference.py --device gpu
# python inference.py --device tpu
# import the necessary packages
from Vinit.data_preprocess import load_dataset
from Vinit.utils import zoom_into_images
from Vinit import config
from tensorflow import distribute
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.io.gfile import glob
from matplotlib.pyplot import subplots
import argparse
import sys
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, default="gpu",
    choices=["gpu", "tpu"], type=str,
    help="device to use for training (gpu or tpu)")
args = vars(ap.parse_args())

# check if we are using TPU, if so, initialize the strategy
# accordingly
if args["device"] == "tpu":
    # initialize the tpus
    tpu = distribute.cluster_resolver.TPUClusterResolver()
    experimental_connect_to_cluster(tpu)
    initialize_tpu_system(tpu)
    strategy = distribute.TPUStrategy(tpu)
    # ensure the user has entered a valid gcs bucket path
    if config.TPU_BASE_TFR_PATH == "gs://<PATH_TO_GCS_BUCKET>/tfrecord":
        print("[INFO] not a valid GCS Bucket path...")
        sys.exit(0)
    # set the train TFRecords, pretrained generator, and final
    # generator model paths to be used for TPU training
    tfrTestPath = config.TPU_DIV2K_TFR_TEST_PATH
    pretrainedGenPath = config.TPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.TPU_GENERATOR_MODEL
# otherwise, we are using multi/single GPU so initialize the mirrored
# strategy
elif args["device"] == "gpu":
    # define the multi-gpu strategy
    strategy = distribute.MirroredStrategy()
    # set the train TFRecords, pretrained generator, and final
    # generator model paths to be used for GPU training
    tfrTestPath = config.GPU_DIV2K_TFR_TEST_PATH
    pretrainedGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.GPU_GENERATOR_MODEL
# else, invalid argument was provided as input
else:
    # exit the program
    print("[INFO] please enter a valid device argument...")
    sys.exit(0)

# get the dataset
print("[INFO] loading the test dataset...")
testTfr = glob(tfrTestPath + "/*.tfrec")
testDs = load_dataset(testTfr, config.INFER_BATCH_SIZE, train=False)
# get the first batch of testing images
(lrImage, hrImage) = next(iter(testDs))

# call the strategy scope context manager
with strategy.scope(): 
    # load the SRGAN trained models
    print("[INFO] loading the pre-trained and fully trained SRGAN model...")
    srganPreGen = load_model(pretrainedGenPath, compile=False)
    srganGen = load_model(genPath, compile=False)
    
    # predict using SRGAN
    print("[INFO] making predictions with pre-trained and fully trained SRGAN model...")
    srganPreGenPred = srganPreGen.predict(lrImage)
    srganGenPred = srganGen.predict(lrImage)

# plot the respective predictions
print("[INFO] plotting the SRGAN predictions...")
(fig, axes) = subplots(nrows=config.INFER_BATCH_SIZE, ncols=4,
    figsize=(50, 50))

# plot the predicted images from low res to high res
for (ax, lowRes, srPreIm, srGanIm, highRes) in zip(axes, lrImage,
        srganPreGenPred, srganGenPred, hrImage):
    # plot the low resolution image
    ax[0].imshow(array_to_img(lowRes))
    ax[0].set_title("Low Resolution Image")
    # plot the pretrained SRGAN image
    ax[1].imshow(array_to_img(srPreIm))
    ax[1].set_title("SRGAN Pretrained")
    # plot the SRGAN image
    ax[2].imshow(array_to_img(srGanIm))
    ax[2].set_title("SRGAN")
    # plot the high resolution image
    ax[3].imshow(array_to_img(highRes))
    ax[3].set_title("High Resolution Image")

# check whether output image directory exists, if it doesn't, then
# create it
if not os.path.exists(config.BASE_IMAGE_PATH):
    os.makedirs(config.BASE_IMAGE_PATH)

# serialize the results to disk
print("[INFO] saving the SRGAN predictions to disk...")
fig.savefig(config.GRID_IMAGE_PATH)

# plot the zoomed in images
zoom_into_images(srganPreGenPred[0], "SRGAN Pretrained")
zoom_into_images(srganGenPred[0], "SRGAN")
