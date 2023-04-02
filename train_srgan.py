# USAGE
# python train_srgan.py --device tpu
# python train_srgan.py --device gpu
# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
tf.random.set_seed(42)
# import the necessary packages
from Vinit.data_preprocess import load_dataset
from Vinit.srgan import SRGAN
from Vinit.vgg import VGG
from Vinit.srgan_training import SRGANTraining
from Vinit import config
from Vinit.losses import Losses
from tensorflow import distribute
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.keras.optimizers import Adam
from tensorflow.io.gfile import glob
import argparse
import sys
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, default="gpu",
    choices=["gpu", "tpu"], type=str,
    help="device to use for training (gpu or tpu)")
args = vars(ap.parse_args())

# check if we are using TPU, if so, initialize the TPU strategy
if args["device"] == "tpu":
    # initialize the TPUs
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
    tfrTrainPath = config.TPU_DIV2K_TFR_TRAIN_PATH
    pretrainedGenPath = config.TPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.TPU_GENERATOR_MODEL
# otherwise, we are using multi/single GPU so initialize the mirrored
# strategy
elif args["device"] == "gpu":
    # define the multi-gpu strategy
    strategy = distribute.MirroredStrategy()
    # set the train TFRecords, pretrained generator, and final
    # generator model paths to be used for GPU training
    tfrTrainPath = config.GPU_DIV2K_TFR_TRAIN_PATH
    pretrainedGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.GPU_GENERATOR_MODEL
# else, invalid argument was provided as input
else:
    # exit the program
    print("[INFO] please enter a valid device argument...")
    sys.exit(0)

# display the number of accelerators
print("[INFO] number of accelerators: {}..."
    .format(strategy.num_replicas_in_sync))
# grab train TFRecord filenames
print("[INFO] grabbing the train TFRecords...")
trainTfr = glob(tfrTrainPath +"/*.tfrec")
# build the div2k datasets from the TFRecords
print("[INFO] creating train and test dataset...")
trainDs = load_dataset(filenames=trainTfr, train=True,
    batchSize=config.TRAIN_BATCH_SIZE * strategy.num_replicas_in_sync)

# call the strategy scope context manager
with strategy.scope():
    # initialize our losses class object
    losses = Losses(numReplicas=strategy.num_replicas_in_sync)
    # initialize the generator, and compile it with Adam optimizer and
    # MSE loss
    generator = SRGAN.generator(
        scalingFactor=config.SCALING_FACTOR,
        featureMaps=config.FEATURE_MAPS,
        residualBlocks=config.RESIDUAL_BLOCKS)
    generator.compile(
        optimizer=Adam(learning_rate=config.PRETRAIN_LR),
        loss=losses.mse_loss)
    # pretraining the generator
    print("[INFO] pretraining SRGAN generator...")
    generator.fit(trainDs, epochs=config.PRETRAIN_EPOCHS,
        steps_per_epoch=config.STEPS_PER_EPOCH)

# check whether output model directory exists, if it doesn't, then
# create it
if args["device"] == "gpu" and not os.path.exists(config.BASE_OUTPUT_PATH):
    os.makedirs(config.BASE_OUTPUT_PATH)

# save the pretrained generator
print("[INFO] saving the SRGAN pretrained generator to {}..."
    .format(pretrainedGenPath))
generator.save(pretrainedGenPath)

# call the strategy scope context manager
with strategy.scope():
    # initialize our losses class object
    losses = Losses(numReplicas=strategy.num_replicas_in_sync)
    # initialize the vgg network (for perceptual loss) and discriminator
    # network
    vgg = VGG.build()
    discriminator = SRGAN.discriminator(
        featureMaps=config.FEATURE_MAPS, 
        leakyAlpha=config.LEAKY_ALPHA, discBlocks=config.DISC_BLOCKS)
    # build the SRGAN training model and compile it
    srgan = SRGANTraining(
        generator=generator,
        discriminator=discriminator,
        vgg=vgg,
        batchSize=config.TRAIN_BATCH_SIZE)
    srgan.compile(
        dOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        gOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        bceLoss=losses.bce_loss,
        mseLoss=losses.mse_loss,
    )
    # train the SRGAN model
    print("[INFO] training SRGAN...")
    srgan.fit(trainDs, epochs=config.FINETUNE_EPOCHS,
        steps_per_epoch=config.STEPS_PER_EPOCH)

# save the SRGAN generator
print("[INFO] saving SRGAN generator to {}...".format(genPath))
srgan.generator.save(genPath)
