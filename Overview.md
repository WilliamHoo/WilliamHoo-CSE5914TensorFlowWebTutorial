# Overview

## Installing

Tensorflow is fully supported in python and has a C API. There are also APIs for Java and Go, but they do not support backwards compatiblity.

Can be installed using pip:
    pip3 install —upgrade tensorflow

Other installing processes for languages and operating systems can be found here: https://www.tensorflow.org/install/

## What is tensor flow

Tensor flow is a computational graph based library. The library supports creation of graphs, training of the graphs, and running the graph. 

A rough application may be something as such:

Where x and y are input data, w are weights for the associated input data and z is the result of such data. Of course realistic applications of tensor flow will likely have hundreds or thousands of such nodes. 

Training of data is often in terms of the weights provided above, with its objective to minimize the 'loss' of the output, z. The 'loss' is specified in terms of a loss function which is discussed later.

One a graph has been fit, then it can be used to predict output of unseen data!

## Creation of graphs

Creation of the graphs is simple as it sounds! It involves creating vetices and edges and the relationships between them. 

To begin, first nodes must be created. Here is an example of creating two nodes:

    :::python
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly

These nodes have 'input' values of constant values. There are a various number of types of input data, for instance is is possible to have the values of such nodes be input data that varies between data.

Next these nodes can be combined in various ways, for instance is is possible to add the values of these nodes. Here is an example of creating a new two that is the addition of node1 and node2:

    :::python
    node3 = tf.add(node1, node2)

This is an example of a very simple graph. Of course this example can be expanded to much more complicated graphs as there are many different operations that are avaiable with TensorFlow. For very complicated graphs it also may be infesible to create a graph as described above. Later, we talk about easier methods to create more complex structures.

## Training of the Graphs

The goal of training is to find a graph (neural network) that produces an expected output given some input. A loss function describes how well the graph matches the 'ground truth'. In order to find the best variables the loss function is minimized. Tensor flow provides both loss functions and optimizers.

## What is tensor flow used for

Tensor flow has a large number of applications such as image recognition, speech recognition, natural language processing and much much more. Most of these applications also require data additional processing to get data into a format applicable for tensor flow. 

TensorFlow walks through an example of using TensorFlow to recognize handwritten digits. Of course this is just one example and there are many other applications of TensorFlow.

## Data imports

With more complicated models and data it may not be feasible to go throughout the entire process in code. Also it is possible that training data may change frequently. In these cases it may be desirable to use TensorFlow functionality to import data from inputfiles.

TensowFlow uses NumPy frequently to load and manipulate data. More information on NumPy can be found here: http://www.numpy.org/.

Input data can be stored using NumPy, TFRecord, and in text data. An example of loading in data stored in NumPy is as follows:

    :::python
    \# Load the training data into two NumPy arrays, for example using `np.load()`.
    with np.load("/var/data/training_data.npy") as data:
      features = data["features"]
      labels = data["labels"]

    \# Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

Features within this context refers to 'input' data and 'labels' is output data. This will create a data set that can then be used to train TensorFlow to to be used to run an existing TensorFlow to see how well it performs on unseen data.
