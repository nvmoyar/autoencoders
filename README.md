# Autoencoders

A simple and a convolutional autoencoder to compress the MNIST dataset. Autoencoders are a type of neural network architecture to perform data compression where the compression and decompression functions are learned from the data itself. Encoder and decoder are both neural networks. Even though autoencoders are not really better than the hand-engineering compression methods in compression, they are really good for image denoising and dimensionality reduction. 

All these experiments are built with TensorFlow. Udacity's original repo can be found [here](http://github.com/deep-learning/autoencoder/). Initially the exercise proposed was a [simple autoencoder](Simple_Autoencoder.ipynb)using FF neural network, and [convolutional autoencoder](Convolutional_Autoencoder_MNIST.ipynb). The idea of the third notebook, another [convolutional autoencoder](Convolutional_Autoencoder_fashionMNIST.ipynb), was to learn how to train this kind of architecture in a bit more complicated dataset -compared to MNIST- like [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), a dataset created and maintained by Zalando. 

### Test and Demo

* [Test](http://localhost:8888/notebooks/autoencoders/)
* [Demo](https://www.floydhub.com/nvmoyar/projects/fashion-mnist-autoencoder)

#### Requirements

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance on the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with Floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use Floyd info XXXXXXXXXXXXXXXXXXXXXX

#### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

#### Datasets 

With FloydHub, you are uploading data from your machine to their remote instance. It's a really bad idea to upload large datasets like CIFAR along with your scripts. Instead, you should always download the data on the FloydHub instance instead of uploading it from your machine.

Further Reading: [How and Why mount data to your job](https://docs.floydhub.com/guides/data/mounting_data/)

### Usage 

floyd run --gpu --env tensorflow-1.2 --message 'Update README' --data floydhub/datasets/mnist/1:mnist --mode jupyter

[**You only need to mount the data to your job, since the dataset has been already been uploaded for you**]

#### Output

Often you'll be writing data out, things like TensorFlow checkpoints, updated notebooks, trained models and HDF5 files. You will find all these files, you can get links to the data with:

> floyd output run_ID