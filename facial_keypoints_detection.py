import tensorflow as tf
import numpy as np
import time
import sys
from util import *

"""
Train a CNN w/dropout and adam optimizer for the facial keypoints detection challenge on Kaggle.
Requires training.csv and test.csv from the challenge site: https://www.kaggle.com/c/facial-keypoints-detection

author: Noah Golmant
"""

def weight_variable(shape, sdev=0.1):
    """ randomly initialize a weight variable w/given shape """
    initial = tf.truncated_normal(shape, stddev=sdev, seed=SEED)
    return tf.Variable(initial)


def bias_variable(shape, constant=0.1):
    """ initialize bias to constant vector w/given shape """
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)


def error_measure(predictions, labels):
    """ calculate sum squared error of predictions """
    return np.sum(np.power(predictions - labels, 2)) / (2.0 * predictions.shape[0])


def eval_in_batches(data, sess, eval_prediction, eval_data_node):
    """
    Evaluate the data on the eval_prediction NN model in batches
    
    data: (num_images, img_size, img_size, num_channels) image data tensor
    sess: tensorflow session
    eval_prediction: neural network model to predict from
    eval_data_node:  model input placeholder for feed_dict

    returns (num_images, num_labels) shaped tensor of prediction vectors for each data image
    """
    size = data.shape[0] # num images in data
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset size: %d" % size)
    
    predictions = np.ndarray(shape = (size, NUM_LABELS), dtype = np.float32)
    for begin in range(0, size, EVAL_BATCH_SIZE): 
        end = begin + EVAL_BATCH_SIZE
        # get next batch from begin index to end index
        if end <= size:
            predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict = { eval_data_node: data[begin:end, ...] })
        else: # if end index is past the end of the data, fit input to batch size required for feed_dict
            batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict = { eval_data_node: data[-EVAL_BATCH_SIZE:, ...] })
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


if __name__ == '__main__':
    dprint("Loading training set from training.csv...")
    train_data, train_labels = load('training.csv')
    dprint("Loading test set from test.csv")
    test_data, _ = load('test.csv')

    dprint("Creating validation data.")
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]

    train_size = train_labels.shape[0]
    dprint("%d training samples." % train_size)

    dprint("Constructing tensorflow models...")
    train_data_node = tf.placeholder(
            tf.float32,
            shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    # input node to training CNN
    train_labels_node = tf.placeholder(tf.float32, shape = (BATCH_SIZE, NUM_LABELS))

    # input node to evaluation CNN
    eval_data_node = tf.placeholder(
            tf.float32,
            shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # convolution layer 1 (1 -> 32 feature maps over 5x5 filter)
    conv1_weights = weight_variables([5, 5, NUM_CHANNELS, 32])
    conv1_biases  = bias_variable([32])

    # convolution layer 2 (32 maps -> 64 maps over 5x5 filter)
    conv2_weights = weight_variable([5, 5, 32, 64])
    conv2_biases = bias_variable([64])

    # fully connected hidden layer (to 512 units)
    fc1_weights = weight_variable([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512])
    fc1_biases = bias_variable([512])

    # 2nd hidden layer
    fc2_weights = weight_variable([512, 512])
    fc2_biases = bias_variable([512])

    # output layer
    fc3_weights = weight_variable([512, NUM_LABELS])
    fc3_biases = bias_variable([NUM_LABELS])

    def model(data, train=False):
        """ Construct a CNN given the data graph node and the above network weights """
        # convolution w/stride 1
        conv = tf.nn.conv2d(data, conv1_weights, strides = [1, 1, 1, 1], padding='SAME')
        
        # bias + ReLU unit
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

        # max pooling (window of 2x2, stride 2)
        pool = tf.nn.max_pool(relu, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')

        # second conv layer bias/relu
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # reshape feature map to feed image vectors to fully connected layers
        # filter_channels x final_image_size x final_image_size shape
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
                pool,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        # fully connected hidden layer
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        # add training dropout
        if train:
            hidden = tf.nn.dropout(hidden, DROPOUT_RATE, seed=SEED)
        hidden = tf.nn.relu(tf.matmul(hidden, fc2_weights) + fc2_biases)

        if train:
            hidden = tf.nn.dropout(hidden, DROPOUT_RATE, seed=SEED)

        return tf.matmul(hidden, fc3_weights) + fc3_biases
    
    # construct training model
    train_prediction = model(train_data_node, True)
    
    # minimize squared err per sample
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(train_prediction - train_labels_node), 1))

    # L2 regularizers for fully connected layers (shared parameter regularization)
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) + 
                    tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))

    loss += L2_REG_PARAM * regularizers

    # eval model for prediction batches
    eval_prediction = model(eval_data_node)

    # set up step increment to calculate learning rate
    global_step = tf.Variable(0, trainable = False)

    # decay learning rate exponentially each epoch
    learning_rate = tf.train.exponential_decay(
            LEARNING_BASE_RATE,
            global_step * BATCH_SIZE, # current image index in dataset
            train_size, # decay step
            LEARNING_DECAY_RATE,
            staircase = True)
    
    train_step = tf.train.AdamOptimizer(learning_rate, ADAM_REG_PARAM).minimize(loss, global_step=global_step)

    # start tensorflow session
    dprint("Starting tensorflow session...")
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    # keep track of loss for training/validation data
    loss_train_record = list()
    loss_valid_record = list()

    start_time = time.gmtime()

    # keep track of time since a major improvement in case we want to stop early
    best_valid = np.inf
    best_valid_epoch = 0
    current_epoch = 0

    dprint("Started training session.")
    while current_epoch < NUM_EPOCHS:
        # shuffle training data
        shuffled_indices = np.arange(train_size)
        np.random.shuffle(shuffled_indices)
        train_data = train_data[shuffled_indices]
        train_labels = train_labels[shuffled_indices]

        # get one batch per step
        for step in range(train_size // BATCH_SIZE):
            offset = step * BATCH_SIZE
            # get batch slice for the step
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # feed batch to training model
            feed_dict = { 
                    train_data_node: batch_data, 
                    train_labels_node: batch_labels
            }
            _, loss_train, current_learning_rate = sess.run(
                    [train_step, loss, learning_rate],
                    feed_dict = feed_dict)
        # evaluate this epoch
        eval_result = eval_in_batches(validation_data, sess, eval_prediction, eval_data_node)
        loss_valid = error_measure(eval_result, validation_labels)

        dprint("Epoch %04d, train loss %.8f, validation loss %.8f, train/validation %0.8f, learning rate %0.8f" %
                (current_epoch, loss_train, loss_valid, loss_train / loss_valid, current_training_rate))
        
        loss_train_record.append(np.log10(loss_train))
        loss_valid_record.append(np.log10(loss_valid))

        sys.stdout.flush()

        # patience check to end early if we can
        if loss_valid < best_valid:
            best_valid = loss_valid
            best_valid_epoch = current_epoch
        elif best_valid_epoch + EARLY_STOP_PATIENCE < current_epoch:
            dprint("Early stopping.")
            dprint("Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch))
            break
        
        current_epoch += 1

    dprint("training finished")
    end_time = time.gmtime()

    dprint(time.strftime('%H:%M:%S', start_time))
    dprint(time.strftime('%H:%M:%S', end_time))

    dprint("Generating test results.")
    generate_test_results(test_data, sess, eval_prediction, eval_data_node)
