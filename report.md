# Part 3

This section involves looking at implementing the `learn_ROI.py` file.

## Part 3, Question 1

<!-- Among other comments that you find interesting, report the initial architecture used (num- ber of layers, number of neurons per layer, activation function etc. ) and the obtained perfor- mance. -->

For this section, `PyTorch` was used 

We do not make use of the preprocessor as the values were relatively bounded enough, and the model being sophisticated enough to deal with the raw dataset.


defaults = {
    'l1':60, 
    'l2':35, 
    'l3':20, 
    'l4':6,
    'lr':1e-3,
    'save':False,
    'epochs':20,
    'filepath': './Model/roi_model.pt'
## Part 3, Question 2

## Part 3, Question 3

Fine tuning uses a similar method to as discussed in Part 2. A bayesian optimisation approach was used on determining the optimal architecture of the network, and the starting learning rate of our optimisation function. 