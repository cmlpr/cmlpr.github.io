---
layout: post
title: "Classifying Traffic Signs"
date: 2017-01-09
mathjx: true
hlighters: htmlcode
image: traffic_signs.jpg
imagealt: "Traffic Sign Classification"
description: "Building a Convolutional Neural Network (CNN) model to classify traffic signs"
category: Neural Networks
tags: tensorflow neural_networks deep_neural_networks convolutional_neural_networks python opencv image_processing self_driving_cars matplotlib sklearn
published: true
comments: true
---


More explanation to follow...

<!--more-->

All files including the Ipython notebook solution and the results are located in this **[repository](https://github.com/cmlpr/CarND-LaneLines-P1)**.

## Helper Functions
***


```python
# Importing Various Packages
########################################################################

import sys
import csv
import pprint
import pickle
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import flatten
```

```python
# Pre-process images
########################################################################

def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')
    :param img: image to be converted to grayscale
    :return: image in grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def hist_equalize(img):
    """
    Improve the contrast of image
    Helps distribute the range of color in the image
    Read more at
    http://docs.opencv.org/trunk/d5/daf/tutorial_py_histogram_equalization.html
    :param img:
    :return:
    """
    return cv2.equalizeHist(img)


def normalize_scale(img):
    """

    :param img:
    :return:
    """
    normalized_image = np.divide(img, 255.0)
    return normalized_image


def pre_process_images(img_list):

    count = len(img_list)
    shape = img_list[0].shape
    processed = []
    for i in range(count):
        img = normalize_scale(hist_equalize(grayscale(img_list[i])))
        processed.append(img)

    return np.reshape(np.array(processed), [count, shape[0], shape[1], 1])
```

```python
# Read and manipulate data
########################################################################

def get_data(valid_set_frac, params):

    # CSV file for sign names
    names_file = "traffic_signs/raw_data/signnames.csv"

    # Folders for pickled training and test images
    training_file = "traffic_signs/raw_data/train.p"
    testing_file = "traffic_signs/raw_data/test.p"

    # Load class names into dictionary
    with open(names_file, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # skip the header
        signnames = {rows[0]: rows[1] for rows in reader}

    # Load images
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    # Assign data as model and test
    x_model, y_model = train['features'], train['labels']
    x_test, y_test = test['features'], test['labels']

    # Shuffle data used in model training
    x_model, y_model = shuffle(x_model, y_model)

    # Make sure the x and y data have same length
    assert (len(x_model) == len(y_model))
    assert (len(x_test) == len(y_test))

    # Image pre-processing
    if params['pre-process']:
        x_model = pre_process_images(x_model)
        x_test = pre_process_images(x_test)
        # print(len(x_model))
        # print(type(x_model))
        # print(x_model.shape)
        # print(np.max(x_model[0]), np.min(x_model[0]))
        # sys.exit()
    else:
        pass

    if 0.05 <= valid_set_frac <= 0.3:
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_model,
            y_model,
            test_size=valid_set_frac,
            random_state=832289)
    else:
        sys.exit("Validation set size should be between 0.05 and 0.3")

    n_train = len(x_train)
    n_valid = len(x_valid)
    n_test = len(x_test)
    n_classes = len(set(y_train))

    print("\nImage Shape: {}\n".format(x_train[0].shape))
    print("Training Set:   {} samples".format(n_train))
    print("Validation Set: {} samples".format(n_valid))
    print("Test Set:       {} samples\n".format(n_test))
    print("Number of Classes: {}\n".format(n_classes))

    unique, counts = np.unique(y_train, return_counts=True)
    class_info = {}
    for unique, counts in zip(unique, counts):
        class_info[str(unique)] = {'description': signnames[str(unique)], 'count': counts}

    pp = pprint.PrettyPrinter(indent=4, width=100)
    print("Example Count in Each Class\n")
    pp.pprint(class_info)

    return x_train, x_valid, x_test, y_train, y_valid, y_test, class_info
```

```python
# Visualize Data
########################################################################

def visualize_data(x, y, class_info, params):

    # Plot the first image
    plt.figure(1)
    if params['pre-process']:
        plt.imshow(x[0].squeeze(), cmap='gray')
    else:
        plt.imshow(x[0])
    plt.title(class_info[str(y[0])]['description'])
    plt.savefig('traffic_signs/images/first_image')
    print('\nCompleted plotting the first image')

    # Plot random 16 images
    plt.figure(2)
    n_train = len(x)
    grid = np.random.randint(n_train, size=(4, 4))
    fig, axes = plt.subplots(4, 4, figsize=(8, 8),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.2, wspace=0.05)
    fig.suptitle('Random 16 images', fontsize=20)
    for ax, i in zip(axes.flat, list(grid.reshape(16, 1))):
        if params['pre-process']:
            ax.imshow(x[int(i)].squeeze(), cmap='gray')
        else:
            ax.imshow(x[int(i)])
        title = str(i) + " - " + class_info[str(y[int(i)])]['description']
        ax.set_title(title, fontsize=8)

    plt.savefig('traffic_signs/images/16_random_images')
    plt.close()
    print('\nCompleted plotting the random 16 images')

    # Bar plot showing the count of each class
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(3)
    plt.bar(unique, counts, 0.5, color='b')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Class')
    plt.savefig('traffic_signs/images/class_freq_plot')
    print('\nCompleted plotting class frequency bar plot')
```

## Model Architecture
***

```python
# Implement Lenet based CNN model
########################################################################

def lenet_model1(data_x, params, channel_count, keep_prob):

    # The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels.

    # Architecture
    # Layer 1: Convolutional. The output shape is 28x28x6.
    # Activation function
    # Pooling. The output shape is 14x14x6.
    # Layer 2: Convolutional. The output shape should be 10x10x16.
    # Activation function
    # Pooling. The output shape is 5x5x16.
    # Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
    # The easiest way to do is by using tf.contrib.layers.flatten
    # Layer 3: Fully Connected. This has 120 outputs.
    # Activation function
    # Layer 4: Fully Connected. This has 84 outputs.
    # Activation function
    # Layer 5: Fully Connected (Logits). 43 class outputs.
    # Output
    # Return the result of the 2nd fully connected layer.

    # Hyperparameters
    mu = params['mean']
    sigma = params['std']
    chn = channel_count

    layer_depth = {
        'conv_1': 6,
        'conv_2': 16,
        'full_1': 120,
        'full_2': 84,
        'out': params['class_count']
    }

    # Store layers weight & bias
    weights = {
        'conv_1': tf.Variable(tf.truncated_normal([5, 5, chn, layer_depth['conv_1']], mean=mu, stddev=sigma)),
        'conv_2': tf.Variable(tf.truncated_normal([5, 5, layer_depth['conv_1'], layer_depth['conv_2']],
                                                  mean=mu, stddev=sigma)),
        'full_1': tf.Variable(tf.truncated_normal([5 * 5 * 16, layer_depth['full_1']], mean=mu, stddev=sigma)),
        'full_2': tf.Variable(tf.truncated_normal([layer_depth['full_1'], layer_depth['full_2']],
                                                  mean=mu, stddev=sigma)),
        'out':    tf.Variable(tf.truncated_normal([layer_depth['full_2'], layer_depth['out']],
                                                  mean=mu, stddev=sigma))
    }
    biases = {
        'conv_1': tf.Variable(tf.zeros(layer_depth['conv_1'])),
        'conv_2': tf.Variable(tf.zeros(layer_depth['conv_2'])),
        'full_1': tf.Variable(tf.zeros(layer_depth['full_1'])),
        'full_2': tf.Variable(tf.zeros(layer_depth['full_2'])),
        'out':    tf.Variable(tf.zeros(layer_depth['out']))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = tf.nn.conv2d(data_x, weights['conv_1'], strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, biases['conv_1'])

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    if params['dropout']:
        conv1 = tf.nn.dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = tf.nn.conv2d(conv1, weights['conv_2'], strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, biases['conv_2'])

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    if params['dropout']:
        conv2 = tf.nn.dropout(conv2, keep_prob)

    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(fc1, weights['full_1']), biases['full_1'])

    # Activation.
    fc1 = tf.nn.relu(fc1)
    if params['dropout']:
        fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1, weights['full_2']), biases['full_2'])

    # Activation.
    fc2 = tf.nn.relu(fc2)
    if params['dropout']:
        fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    if params['l2_reg']:
        reg_term = params['l2_beta'] * (tf.nn.l2_loss(weights['conv_1']) + tf.nn.l2_loss(weights['conv_2']) +
                                        tf.nn.l2_loss(weights['full_1']) + tf.nn.l2_loss(weights['full_2']))
    else:
        reg_term = None

    return logits, reg_term
```

## Model Pipeline
***

```python
# Model Pipeline
########################################################################

def run_model(train_x, train_y, valid_x, valid_y, test_x, test_y, params):

    model_name = params['name']

    channel_count = train_x[0].shape[2]

    # Placeholder for batch of input images
    model_x = tf.placeholder(tf.float32, (None, 32, 32, channel_count))
    # Placeholder for batch of output labels
    model_y = tf.placeholder(tf.int32, None)
    one_hot_y = tf.one_hot(model_y, params['class_count'])
    # Dropout only
    keep_prob = tf.placeholder(tf.float32)
    result_logits, reg_adder = lenet_model1(model_x, params, channel_count, keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(result_logits, one_hot_y)
    if params['l2_reg']:
        loss_operation = tf.reduce_mean(cross_entropy) + reg_adder
    else:
        loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['rate'])
    training_operation = optimizer.minimize(loss_operation)

    # Model evaluation
    correct_prediction = tf.equal(tf.argmax(result_logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Save
    saver = tf.train.Saver()

    if params['action'].upper() == 'TRAIN':

        # Run the training data through the pipeline to train the model
        # Before each epoch, shuffle the training set
        # After each epoch, measure the loss and accuracy on the validation set
        # Save the model after training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(train_x)
            print("\nTraining Model: {}\n".format(model_name))
            for i in range(params['epoch']):
                train_x, train_y = shuffle(train_x, train_y)
                for offset in range(0, num_examples, params['batch_size']):
                    end = offset + params['batch_size']
                    batch_x, batch_y = train_x[offset:end], train_y[offset:end]
                    if params['dropout']:
                        sess.run(training_operation, feed_dict={model_x: batch_x, model_y: batch_y,
                                                                keep_prob: params['dropout_prob']})
                    else:
                        sess.run(training_operation, feed_dict={model_x: batch_x, model_y: batch_y})

                num_valid_examples = len(valid_x)
                total_accuracy = 0
                for offset2 in range(0, num_valid_examples, params['batch_size']):
                    batch_valid_x, batch_valid_y = valid_x[offset2:offset2 + params['batch_size']], \
                                                   valid_y[offset2:offset2 + params['batch_size']]
                    if params['dropout']:
                        accuracy = sess.run(accuracy_operation,
                                            feed_dict={model_x: batch_valid_x, model_y: batch_valid_y, keep_prob: 1.0})
                    else:
                        accuracy = sess.run(accuracy_operation,
                                            feed_dict={model_x: batch_valid_x, model_y: batch_valid_y})
                    total_accuracy += (accuracy * len(batch_valid_x))
                validation_accuracy = total_accuracy / num_valid_examples

                print("EPOCH {}: Validation Accuracy = {:.3f}".format(i + 1, validation_accuracy))

            saver.save(sess, 'traffic_signs/models/' + model_name)

    elif params['action'].upper() == 'TEST':

        print("\nTesting Model: {}\n".format(model_name))

        load_file = 'traffic_signs/models/' + model_name

        with tf.Session() as sess:
            saver.restore(sess, load_file)

            num_test_examples = len(test_x)
            total_accuracy = 0
            for offset in range(0, num_test_examples, params['batch_size']):
                batch_test_x, batch_test_y = test_x[offset:offset + params['batch_size']], \
                                             test_y[offset:offset + params['batch_size']]
                if params['dropout']:
                    accuracy = sess.run(accuracy_operation,
                                        feed_dict={model_x: batch_test_x, model_y: batch_test_y, keep_prob: 1.0})
                else:
                    accuracy = sess.run(accuracy_operation,
                                        feed_dict={model_x: batch_test_x, model_y: batch_test_y})
                total_accuracy += (accuracy * len(batch_test_y))
            test_accuracy = total_accuracy / num_test_examples

            print("Test Accuracy = {:.3f}".format(test_accuracy))
    else:
        sys.exit("Action can be train or test only")
```

## Training the model
***

```python
# Run model
########################################################################

def run():

    pre_process_param = {
        'pre-process': True
    }

    x_train, x_valid, x_test, y_train, y_valid, y_test, class_info = get_data(0.2, pre_process_param)
    visualize_data(x_train, y_train, class_info, pre_process_param)

    model_param_list = {
        'action': 'train',
        'name': 'trial_12',
        'epoch': 30,
        'batch_size': 128,
        'mean': 0.,
        'std': 0.1,
        'class_count': len(class_info),
        'rate': 0.001,
        'l2_reg': True,ß
        'l2_beta': 0.01,
        'dropout': True,
        'dropout_prob': 0.9
    }

    run_model(x_train, y_train, x_valid, y_valid, x_test, y_test, model_param_list)


run()
```

Training Output: 

```
Image Shape: (32, 32, 1)

Training Set:   31367 samples
Validation Set: 7842 samples
Test Set:       12630 samples

Number of Classes: 43

Example Count in Each Class

{   '0': {'count': 172, 'description': 'Speed limit (20km/h)'},
    '1': {'count': 1777, 'description': 'Speed limit (30km/h)'},
    '10': {'count': 1603, 'description': 'No passing for vehicles over 3.5 metric tons'},
    '11': {'count': 1050, 'description': 'Right-of-way at the next intersection'},
    '12': {'count': 1655, 'description': 'Priority road'},
    '13': {'count': 1704, 'description': 'Yield'},
    '14': {'count': 617, 'description': 'Stop'},
    '15': {'count': 497, 'description': 'No vehicles'},
    '16': {'count': 321, 'description': 'Vehicles over 3.5 metric tons prohibited'},
    '17': {'count': 888, 'description': 'No entry'},
    '18': {'count': 985, 'description': 'General caution'},
    '19': {'count': 173, 'description': 'Dangerous curve to the left'},
    '2': {'count': 1796, 'description': 'Speed limit (50km/h)'},
    '20': {'count': 286, 'description': 'Dangerous curve to the right'},
    '21': {'count': 252, 'description': 'Double curve'},
    '22': {'count': 307, 'description': 'Bumpy road'},
    '23': {'count': 415, 'description': 'Slippery road'},
    '24': {'count': 228, 'description': 'Road narrows on the right'},
    '25': {'count': 1196, 'description': 'Road work'},
    '26': {'count': 479, 'description': 'Traffic signals'},
    '27': {'count': 199, 'description': 'Pedestrians'},
    '28': {'count': 442, 'description': 'Children crossing'},
    '29': {'count': 217, 'description': 'Bicycles crossing'},
    '3': {'count': 1121, 'description': 'Speed limit (60km/h)'},
    '30': {'count': 377, 'description': 'Beware of ice/snow'},
    '31': {'count': 614, 'description': 'Wild animals crossing'},
    '32': {'count': 206, 'description': 'End of all speed and passing limits'},
    '33': {'count': 549, 'description': 'Turn right ahead'},
    '34': {'count': 323, 'description': 'Turn left ahead'},
    '35': {'count': 967, 'description': 'Ahead only'},
    '36': {'count': 308, 'description': 'Go straight or right'},
    '37': {'count': 169, 'description': 'Go straight or left'},
    '38': {'count': 1662, 'description': 'Keep right'},
    '39': {'count': 249, 'description': 'Keep left'},
    '4': {'count': 1592, 'description': 'Speed limit (70km/h)'},
    '40': {'count': 290, 'description': 'Roundabout mandatory'},
    '41': {'count': 194, 'description': 'End of no passing'},
    '42': {'count': 183, 'description': 'End of no passing by vehicles over 3.5 metric tons'},
    '5': {'count': 1509, 'description': 'Speed limit (80km/h)'},
    '6': {'count': 319, 'description': 'End of speed limit (80km/h)'},
    '7': {'count': 1162, 'description': 'Speed limit (100km/h)'},
    '8': {'count': 1139, 'description': 'Speed limit (120km/h)'},
    '9': {'count': 1175, 'description': 'No passing'}}

Completed plotting the first image
```

![png](/images/traffic_signs/first_image.png)


```
Completed plotting the random 16 images
```

![png](/images/traffic_signs/16_random_images.png)

```
Completed plotting class frequency bar plot
```

![png](/images/traffic_signs/class_freq_plot.png)

```
Training Model: trial_12

EPOCH 1: Validation Accuracy = 0.475
EPOCH 2: Validation Accuracy = 0.695
EPOCH 3: Validation Accuracy = 0.793
EPOCH 4: Validation Accuracy = 0.831
EPOCH 5: Validation Accuracy = 0.856
EPOCH 6: Validation Accuracy = 0.860
EPOCH 7: Validation Accuracy = 0.881
EPOCH 8: Validation Accuracy = 0.879
EPOCH 9: Validation Accuracy = 0.881
EPOCH 10: Validation Accuracy = 0.900
EPOCH 11: Validation Accuracy = 0.903
EPOCH 12: Validation Accuracy = 0.905
EPOCH 13: Validation Accuracy = 0.893
EPOCH 14: Validation Accuracy = 0.899
EPOCH 15: Validation Accuracy = 0.911
EPOCH 16: Validation Accuracy = 0.920
EPOCH 17: Validation Accuracy = 0.927
EPOCH 18: Validation Accuracy = 0.927
EPOCH 19: Validation Accuracy = 0.926
EPOCH 20: Validation Accuracy = 0.912
EPOCH 21: Validation Accuracy = 0.921
EPOCH 22: Validation Accuracy = 0.925
EPOCH 23: Validation Accuracy = 0.924
EPOCH 24: Validation Accuracy = 0.933
EPOCH 25: Validation Accuracy = 0.940
EPOCH 26: Validation Accuracy = 0.936
EPOCH 27: Validation Accuracy = 0.928
EPOCH 28: Validation Accuracy = 0.934
EPOCH 29: Validation Accuracy = 0.939
EPOCH 30: Validation Accuracy = 0.935
```

## Testing the model
***

```python
# Run testing
########################################################################

def run():

    pre_process_param = {
        'pre-process': True
    }

    x_train, x_valid, x_test, y_train, y_valid, y_test, class_info = get_data(0.2, pre_process_param)
    visualize_data(x_train, y_train, class_info, pre_process_param)

    model_param_list = {
        'action': 'test',
        'name': 'trial_12',
        'epoch': 30,
        'batch_size': 128,
        'mean': 0.,
        'std': 0.1,
        'class_count': len(class_info),
        'rate': 0.001,
        'l2_reg': True,
        'l2_beta': 0.01,
        'dropout': True,
        'dropout_prob': 0.9
    }

    run_model(x_train, y_train, x_valid, y_valid, x_test, y_test, model_param_list)


run()
```

Testing Output:

```
Testing Model: trial_12

Test Accuracy = 0.902

Process finished with exit code 0
```