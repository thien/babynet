import numpy as np
from simulator import RobotArm
import torch
import torch.nn.functional as F

area_map = {
    k: v
    for k, v in zip(range(4), ("Zone 1", "Zone 2", "Ground", "Unlabelled area"))
}


def softmax(x):
    numer = np.exp(x - x.max(axis=1, keepdims=True))
    denom = numer.sum(axis=1, keepdims=True)
    return numer / denom


def illustrate_results_ROI(network, nb_pos=10):

    data = (
        (np.random.rand(nb_pos + 1, 7) * 2 - 1) * np.pi / 2
    )  # generating 10 cols to match length of dataset, but only the first 3 are used.
    data[0, :] = 0

    # convert to tensor
    data = torch.Tensor(data)
    # evaluate dataset
    results = network(data[1:, 0:3])
    robot = RobotArm()
    # add back to model
    data[1:, 3:7] = results
    # convert back to numpy
    data = data.detach().numpy()

    prediction = [area_map[x] for x in np.argmax(data[1:, 3:7], axis=1)]

    angles = np.zeros((nb_pos + 1, 6))
    angles[:, 0:3] = data[:, 0:3]
    ax = None
    for i in range(nb_pos):
        ax = robot.animate(angles[i, :], angles[i + 1, :], ax, [0, 0, 0])
        print("Predicted region: {}".format(prediction[i]))


def illustrate_results_FM(network, prep, nb_pos=10):
    data = (
        (np.random.rand(nb_pos + 1, 6) * 2 - 1) * np.pi / 2
    )  # generating 10 cols to match length of dataset, but only the first 3 are used.

    data[0, :] = 0
    data = prep.apply(data)
    x = torch.Tensor(data[1:, 0:3])
    results = network(x)
    results = results.detach().numpy()
    robot = RobotArm()

    data[1:, 3:6] = results
    data = prep.revert(data)

    prediction = data[1:, 3:6]
    angles = np.zeros((nb_pos + 1, 6))
    angles[:, 0:3] = data[:, 0:3]
    ax = None
    for i in range(nb_pos):
        ax = robot.animate(angles[i, :], angles[i + 1, :], ax, prediction[i, :])

if __name__ == "__main__":
    illustrate_results_FM(None)