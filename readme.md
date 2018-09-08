TensorFlow implementation of Polygon RNN++:
1. The Polygon RNN module contains two main models that are trained using policy gradients for the purpose of object annotation: an image representation model and a vertex prediction model.
2. The baseline image representation model is a modified VGG-16 convolutional neural network that takes as input a 224x224x3 RGB image and is also used for predicting the first vertex. This may be improved by using a resnet-50 variant. 
3. The vertex prediction model is a two-layer convolutional LSTM that takes as input a concatenated feature tensor which includes: the CNN feature representation of the image, a one-hot encoding of the previous predicted vertex and the vertex predicted from two time steps ago, as well as the one-hot encoding of the first predicted vertex y1.
4. This set of models can be improved using an attention module, graph neural networks and a value function which tries to estimate the current IoU for the 
current polygon. 
5. For more details concerning the reference paper, I would recommend reading 'Annotating Object Instances with a Polygon-RNN': https://arxiv.org/pdf/1704.05548.pdf as well as: https://arxiv.org/pdf/1803.09693.pdf# vertex_prediction
# vertex_prediction
