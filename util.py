import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from facial_keypoints_detection import eval_in_batches

""" Utils + constants to load data into tensorflow """

# Program constants
NUM_EPOCHS = 1000           # training epoch count
EARLY_STOP_PATIENCE = 100   # epoch count before we stop if no improvement
IMAGE_SIZE = 96             # face images 96x96 pixels
VALIDATION_SIZE = 100       # number of samples in validation dataset
BATCH_SIZE = 64             # number of samples in a training mini-batch
EVAL_BATCH_SIZE = 64        # number of samples in intermediate batch for performance check per epoch
NUM_CHANNELS = 1            # color channels in images
NUM_LABELS = 30             # number of label features per image
SEED = None                 # random seed for weight initialization (None = random)
DEBUG_MODE = True           # print debug lines
DROPOUT_RATE = 0.50         # chance of neuron dropout in training
L2_REG_PARAM = 1e-7         # lambda for L2 regularization of fully connected layer
LEARNING_BASE_RATE = 1e-3   # rates for exponential decay of learning rate
LEARNING_DECAY_RATE = 0.95  # ""
ADAM_REG_PARAM = 0.95       # adam optimizer regularization param


def load(filename, test=False):
    """ 
    Load the facial_keypoints_detection csv data into numpy structures from pandas
    Return (X, y) tuple of (num_images, image_size, image_size, 1) shaped image tensor and (num_samples, num_features) shaped tuple of labels if it is not a test set
    
    """
    dataframe = pd.read_csv(filename)
    feature_cols = dataframe.columns[:-1] # all but image column
    # transform image space-separated pixel values to normalized pixel vector
    dataframe['Image'] = dataframe['Image'].apply(lambda img: np.fromstring(im, sep = ' ') / 255.0)
    dataframe = dataframe.dropna() # drop entries w/NaN entries

    # get all image vectors and reshape to a #num_images x image_size x image_size x channels tensor
    X = np.vstack(dataframe['Image'])
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    if not test:
        # get label features and scale pixel coordinates by image range
        y = dataframe[feature_cols].values / 96.0
        # permute (image, label) pairs for training
        X, y = shuffle(X, y)
    else:
        y = None
    return X, y


def generate_test_results(test_data, sess, eval_prediction, eval_data_node):
    """ Generate predictions for test data on a learned model """
    test_labels = eval_in_batches(test_data, sess, eval_prediction, eval_data_node)
    # rescale predictions to pixel coordinates
    test_labels *= 96.0
    test_labels = test_labels.clip(0, 96)
    
    results = pd.DataFrame(test_labels, columns=('ImageNumber', 'FeatureVector'))
    results.to_csv('result_vectors.csv', index=False)
    dprint("Wrote test results to result_vecotrs.csv.")


def dprint(obj):
    if DEBUG_MODE:
        print(obj)
