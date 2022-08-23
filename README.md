# Digits Recognition on the MNIST dataset using Neural Networks (implemented from scratch)
## The dataset
The dataset used to train the model is the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
## The hierarchy of the project
```bash
Digits Recognition:.
├───model.obj
├───NeuralNetwork.py
├───test.csv
└───train.csv
```
Note that the test.csv, and the train.csv files are compressed into a file called MNIST-dataset.zip with compression level "9 - Ultra" thus it might take sometime to be extracted, the extracted file will be about 128 MB
## Neural Network architecture
The neural network consists of one input layer, 2 hidden layers, and 1 output layer. <br/>
The input layer contains 784 nodes (because the images are all 28 by 28) <br/>
The 1st hidden layer contains 256 nodes <br/>
The 2nd hidden layer contains 128 nodes <br/>
The output layer contains 10 nodes (1 for each digit) <br/>
## The files used in the project
### mode.obj
Contains the weights and biases for a pre trained model, the file is dumped using ```pickle``` it contains a dictionary which looks like this
```python
{
    'w1': w1,
    'b1': b1,
    'w2': w2,
    'b2': b2,
    'w3': w3,
    'b3': b3
}
```
### NeuralNetwork.py
A python file which contains the code for training and predicting the test set, and saving the weights and biases to a file
### test.csv
Contains the testing set as one image per row; the first column contains the label for each row.
### train.csv
Contains the training set as one image per row; the first column contains the label for each row.
