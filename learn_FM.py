import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
import helpers
from nn_lib import Preprocessor
from illustrate import illustrate_results_FM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# create the model.
class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.a = nn.Linear(in_size,128)
        self.d = nn.Linear(128,64)
        self.e = nn.Linear(64,32)
        self.g = nn.Linear(32,8)
        self.f = nn.Linear(8, out_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.a(x))
        x = F.relu(self.d(x))
        x = F.relu(self.e(x))
        x = F.relu(self.g(x))
        # we do not have an activation
        # function on the last layer deliberately.
        x = self.f(x)
        return x
    
    
def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    np.random.shuffle(dataset)
    
    trainRatio = 0.8
    testRatio  = 0.1
    numEpochs  = 30
    batchsize  = 64

    # shuffle the dataset prior
    np.random.shuffle(dataset)

    # preprocess data
    prep = Preprocessor(dataset)
    dataset = prep.apply(dataset)

    # retrieve X and Y columns
    X, Y = dataset[:, 0:3], dataset[:, 3:6]

    # create loaders
    trainloader, validloader, testloader = helpers.loadTrainingData(
        X, Y, trainRatio, testRatio, batchsize)

    # initiate the model
    model = Net(3,3)

    # load hyperparameters
    loss_function = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters())

    # train the model
    helpers.trainModel(model, optimizer, loss_function, numEpochs, trainloader, validloader)
    # evaluate the model
    helpers.testModel(model, loss_function, numEpochs, testloader)
    
    torch.save(model.state_dict(), 'best_model_reg.pth')
    
    x_store, y_store = [],[]
    i =0
    for x,y in testloader:
        if i < 1: 
            x_store = np.array(x)
            y_store = np.array(y)
        else:
            np.concatenate((x_store, np.array(x)), axis = 0)
            np.concatenate((y_store, np.array(y)), axis = 0)
        i = 1

    evaluate_architecture(model, torch.Tensor(x_store), torch.Tensor(y_store), prep)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # data is normalised in this function
    illustrate_results_FM(model, prep)



def evaluate_architecture(model, X, y, prep):
    
    results = model(X)
    results = results.detach().numpy()
    dataset = np.concatenate((X.detach().numpy(), y.detach().numpy()), axis = 1)
    dataset_pred = np.zeros(dataset.shape)
    dataset_pred[:, 3:6] = results
    dataset_pred[:, :3] = dataset[:, :3]
    dataset = prep.revert(dataset)
    dataset_pred = prep.revert(dataset_pred)
    prediction = dataset_pred[:, 3:6]
    true = dataset[:, 3:6]
    
    mae = mean_absolute_error(true, prediction)
    mse = mean_squared_error(true, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, prediction, multioutput = 'uniform_average')
    accuracy = np.average(1 - abs(prediction-true)/1000)
    
    print("\nTest set: Mean absolute error: {:.3f},  R2 score: {:.3f}".format(mae, r2))
    print("\t  Root mean squared error: {:.3f}\n".format(rmse))
    print("\t  Average Accuracy: {:.6f}\n".format(accuracy))
    print("\t  Note: the accuracy is measured as the euclidian distance") 
    print("\t        between true and prediction over a maximum error of 1000")
          
    
    return mae, mse, rmse, r2


def predict_hidden(dataset):
    # -------  Process data -------- #
    prep = Preprocessor(dataset)
    dataset = prep.apply(dataset)
    X, Y = torch.Tensor(dataset[:, 0:3]), torch.Tensor(dataset[:, 3:6])

    # ------- Instantiate model ----- #
    model = Net(3,3)

    # ----- Load our best model ------ #
    model.load_state_dict(torch.load('best_model_reg.pth'))

    # ----- Compute the output ------ #
    results = model(X)
    results = results.detach().numpy()

    # ----- Revert data processing ------ #
    dataset[:, 3:6] = results
    dataset = prep.revert(dataset)
    prediction = dataset[:, 3:6]

    return prediction # Returns a numpy array of shape (n_samples, 3)


if __name__ == "__main__":
    main()
