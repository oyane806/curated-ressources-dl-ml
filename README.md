# Curated list of ressources for deep learning and machine learning - in progress

You can find below a curated list of ressources for deep learning and machine learning. It contains repos, articles, research papers and ebooks. I am trying to keep this list as minimal as possible!

#### python review
a python quick reference :star: [article](https://learnxinyminutes.com/docs/python3/)

#### deep learning ebook
deep learning reference book that you can download for free :star: [ebook](https://github.com/janishar/mit-deep-learning-book-pdf)  
to get familiar with mathematical notations, the hundred pages machine learning book [ebook](https://github.com/ZakiaSalod/The-Hundred-Page-Machine-Learning-Book)

## random forest
Random forests are very good for classification but cannot extrapolate.  
:one: to get a visual intuition of how random forests work [article](http://structuringtheunstructured.blogspot.com/2017/11/coloring-with-random-forests.html)  
:two: to code a random forest from scratch :gear::gear: [notebook](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson3-rf_foundations.ipynb)  
this is a very good exercise to brush up your  coding skills and get an in-depth knowledge of how random forests work  
:three: random forest interpretation :star::star::star: (tree variance feature importance, partial dependence, tree interpreter) [article](https://medium.com/usf-msds/intuitive-interpretation-of-random-forest-2238687cae45)

## neural nets

- **activation functions**: tanh, relu, leaky relu, sigmoid, softmax
- **initialization**: He initialization, Xavier initialization
- **regularization**: L2, L1, dropout, data augmentation, early stopping
- **optimization**: batch gradient descent or vanilla gradient descent, stochastic gradient deschent, mini batch gradient descent, gradient descent with momentum, RMS prop, Adam optimization, learning rate decay, batch normalization

:one: how to use cyclical learning rates for training neural networks :star: [paper](https://arxiv.org/abs/1506.01186)


## time-series
:one: great practical repo to get SOTA results quickly :star::star: [notebooks](https://github.com/timeseriesAI/timeseriesAI)  
:two: quick walk through different time-series predictions methods (neural net, cnn, lstm, cnn-lstm, encoder-decoder lstm) [article](https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/)

#### SOTA algorithms
2017 hive-cote: ensemble of 37 classifiers (no neural net)  
2019 rocket: using random convolutional kernels [paper](https://arxiv.org/pdf/1910.13051)

## image recognition

- filter, padding, stride, convolution, pooling (max, average), fully connected layer

#### cnn architectures

| year | model | size | top-1 accuracy | top-5 accuracy | parameters | depth |
| --- | --- | --- | --- | --- | --- | --- |
| 2012 | [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) |  |  |  |  |  |
| 2014 | [VGG16](https://arxiv.org/abs/1409.1556) | 528MB | 0.713 | 0.901 | 138,357,544 | 23 |
| 2015 | [InceptionV3/GoogleNet](https://arxiv.org/abs/1409.4842) | 92MB | 0.779 | 0.937 | 23,851,784 | 159 |
| 2015 | [ResNet50](https://arxiv.org/abs/1512.03385) :star:| 98MB | 0.749 | 0.921 | 25,636,712 |  |

#### segmentation
2015 unet [paper](https://arxiv.org/abs/1505.04597)
