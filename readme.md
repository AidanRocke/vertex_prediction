TensorFlow implementation of vertex prediction model:
1. The following repository includes a vertex prediction model that may be used to infer mappings from objects in images to the vertices of polygon which bound the objects
of interest. 
2. Using a modified VGG-16 convolutional neural network that takes as input a 224x224x3 RGB image, inference is done in one forward pass that happens in two stages. 
First, an image representation is extracted from blocks of the VGG-16 model. Then a down-sampled variant of this image-representation is passed to a fully-connected layer
which performs a global optimisation step. 
3. For more details concerning the reference paper, I would recommend reading 'Annotating Object Instances with a Polygon-RNN': https://arxiv.org/pdf/1704.05548.pdf as well as: https://arxiv.org/pdf/1803.09693.pdf
