# autoencoders
A simple and a convolutional autoencoder to compress the MNIST dataset.

Autoencoders are a type of neural network architecture to perform data compression where the compression and decompression functions are learned from the data itself.
Encoder and decoder are both neural networks. Even though autoencoders are not really better than the hand-engineering compression
methods in compression, they are really good for image denoising and dimensionality reduction. 

## Requirements

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance in the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use floyd info XXXXXXXXXXXXXXXXXXXXXX

### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

### Datasets 

With FloydHub, you are uploading data from your machine to their remote instance. It's a really bad idea to upload large datasets like CIFAR along with your scripts. Instead, you should always download the data on the FloydHub instance instead of uploading it from your machine.

> floyd run "python train.py" --data diSgciLH4WA7HpcHNasP9j

Here, diSgciLH4WA7HpcHNasP9j is the ID for this dataset, you can get IDs for other data sets from the list linked above. The dataset will be available at /input. So, the CIFAR10 data will be in /input/CIFAR10.

### Output
Often you'll be writing data out, things like TensorFlow checkpoints. Or, updated notebooks. To get these files, you can get links to the data with:

> floyd output run_ID

## Results 

Compare results on training using tf.contrib.layers.fully_connected() vs tf_layers_dense()

Dense/fully connected layer: A linear operation on the layer's input vector.

https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/contrib/layers/fully_connected
https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/layers/dense
