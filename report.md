# Part 3

This section involves looking at implementing the `learn_ROI.py` file.

## Part 3, Question 1

<!-- Among other comments that you find interesting, report the initial architecture used (num- ber of layers, number of neurons per layer, activation function etc. ) and the obtained perfor- mance. -->

For this section, `PyTorch` was used, in a similar fashion to Part 2. We also set the random seed to `1337` to ensure repeatable results.

We do not make use of the preprocessor as the values were relatively bounded enough, and the model is sophisticated enough to deal with the raw dataset. This differs from Part 2 where the prediction was a continuous value, which is naturally more difficult than predicting one-hot encoding of the classes.

Our initial architecture is a neural network with 5 fully connected layers `{20,15,10,7}` with a one-hot output representing the different labels of the dataset. Each layer, other than the output layer has a `ReLU` activation function. We also initiate all the weights of the model with `Xavier`. 

We use the `Adam` optmiser with a starting learning rate of 1e-3, with a Binary Cross Entropy loss function. (In our particular example we use Logits loss which also deals with softmaxing our outputs. This is called with PyTorch's `torch.nn.BCEWithLogitsLoss()`).


## Part 3, Question 2

As we are performing a classification problem instead of a regression problem, alternative measurements are used. We measure the accuracy of the model, alongside Precision, Recall and F1 measures.

After 20 epochs, the model produces the following result:

    Test set: Average loss (on normalised data): 0.0008
    Test Set: Accuracy: 0.969  Precision: 1.000  Recall: 0.969  F1: 0.984


## Part 3, Question 3

Fine tuning uses a similar method to as discussed in Part 2. A bayesian optimisation approach was used on determining the optimal architecture of the network, and the starting learning rate of our optimisation function. 

To use the best model, `predict_hidden(path_of_dataset)` is implemented. This can be loaded into the `__main__` function at the end of `learn_ROI.py`. This will automatically load the best model (pre-trained) and show the performance.