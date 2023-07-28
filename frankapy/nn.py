import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np


class StiffnessEstimator(nn.Module):
    def __init__(self, K_min, K_max, hidden_layers):
        """
                Neural network that will estimate the stiffness values Kp given [xe, ve, he]. The output of the network is
                then computed as a prediction of hc given [Kp, xd, vd]
                Inputs:
                - rotational_stiffness (1x3): Rotational stiffnesses [kx, ky, kz]
                - hidden_layers: hidden layers in Neural Network as a list each element in the list representing the
                                number of neurons in that layer.
                """
        super().__init__()

        self.K_min = K_min
        self.K_max = K_max
        self.hidden_layers = hidden_layers
        Kp_hat_size = 3

        assert len(hidden_layers) >= 1

        self.input_layer = nn.Linear(18, hidden_layers[0])
        nn.init.kaiming_normal_(self.input_layer.weight)

        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)

        self.output_layer = nn.Linear(hidden_layers[-1], Kp_hat_size)
        nn.init.kaiming_normal_(self.output_layer.weight)

    def forward(self, x):
        """
        - x: [xe, ve, he] (batch_size x 18)
        """
        temp_scores = F.relu(self.input_layer(x))
        for layer in self.layers:
            temp_scores = F.relu(layer(temp_scores))
        Kp_trans = torch.sigmoid(self.output_layer(temp_scores))  # "normalized" to range [0,1]
        Kp_trans = Kp_trans * (self.K_max - self.K_min) + self.K_min  # "normalized" to range [K_min, K_max]

        return Kp_trans


class NeuralNet(nn.Module):
    def __init__(self, rotational_stiffness: torch.Tensor, K_min, K_max, hidden_layers):
        """
        Neural network that will estimate the stiffness values Kp given [xe, ve, he]. The output of the network is
        then computed as a prediction of hc given [Kp, xd, vd]
        Inputs:
        - rotational_stiffness (1x3): Rotational stiffnesses [kx, ky, kz]
        - hidden_layers: hidden layers in Neural Network as a list each element in the list representing the
                                number of neurons in that layer.
        - K_min: lower bound of Kp
        - K_max: upper bound of Kp
        """
        super().__init__()
        # self.Kp_rot = rotational_stiffness
        self.K_min = K_min
        self.K_max = K_max
        self.hidden_layers = hidden_layers

        assert len(hidden_layers) >= 1

        self.stiffness_layer = StiffnessEstimator(K_min, K_max, hidden_layers)

    def forward(self, x):
        """
        - x: [xe, ve, he, xd, vd] (batch_size x 30)
        - Only [xe, ve, he] (batch_size x 18) goes through main body of neural network

        Returns: [fc_hat, Kp_trans] each of size (batch_size, 3)
        """

        xe = x[:, :6]
        ve = x[:, 6:12]
        he = x[:, 12:18]
        xd = x[:, 18:24]
        vd = x[:, 24:]

        Kp_trans = self.stiffness_layer(x[:, :18])
        # Kp_rot = self.Kp_rot.repeat(Kp_trans.shape[0], 1)
        # Kp_hat = torch.cat((Kp_trans, Kp_rot), 1)

        fc_hat = 2 * torch.sqrt(Kp_trans) * (vd[:, :3] - ve[:, :3]) + Kp_trans * (xd[:, :3] - xe[:, :3])  # Cartesian coriolis term is asssumed neglectible
        return [fc_hat, Kp_trans]


class StiffnessEstimatorNumpy(nn.Module):
    def __init__(self, K_min, K_max, hidden_layers):
        """
                Neural network that will estimate the stiffness values Kp given [xe, ve, he]. The output of the network is
                then computed as a prediction of hc given [Kp, xd, vd]
                Inputs:
                - rotational_stiffness (1x3): Rotational stiffnesses [kx, ky, kz]
                - hidden_layers: hidden layers in Neural Network as a list each element in the list representing the
                                number of neurons in that layer.
                """
        super().__init__()

        self.K_min = K_min
        self.K_max = K_max
        self.hidden_layers = hidden_layers
        Kp_hat_size = 3

        assert len(hidden_layers) >= 1

        self.input_layer = (np.zeros((hidden_layers[0], 18)), np.zeros((hidden_layers[0])))

        self.layers = []
        for i in range(1, len(hidden_layers)):
            self.layers.append(
                (np.zeros((hidden_layers[i], hidden_layers[i - 1])), np.zeros((hidden_layers[i])))
            )

        self.output_layer = (np.zeros((Kp_hat_size, hidden_layers[-1])), np.zeros((Kp_hat_size)))

    def forward(self, x):
        """
        - x: [xe, ve, he] (batch_size x 18)
        """
        temp_scores = self.input_layer[0] @ x + self.input_layer[1]
        temp_scores = np.clip(temp_scores, 0, 1)
        # temp_scores = F.relu(self.input_layer(x))
        for layer in self.layers:
            temp_scores = layer[0] @ temp_scores + layer[1]
            temp_scores = np.clip(temp_scores, 0, 1)
        Kp_trans = np.clip(self.output_layer[0] @ temp_scores + self.output_layer[1], -24, 24)
        Kp_trans = 1 / (1 + np.exp(-Kp_trans))  # "normalized" to range [0,1]
        Kp_trans = Kp_trans * (self.K_max - self.K_min) + self.K_min  # "normalized" to range [K_min, K_max]

        return Kp_trans


def test_nn():
    input_size = 30
    x = torch.zeros((64, input_size), dtype=torch.float32)  # minibatch size 64, feature dimension 30
    model = NeuralNet(rotational_stiffness=[10., 10., 10.], K_min=0.1, K_max=1000, hidden_layers=[100] * 4)
    #  (rotational_stiffness, K_min, K_max, hidden_layers):
    scores, Kp_t = model(x)
    print(scores.size())  # you should see [64, 6]
    print(Kp_t.size())  # you should see [64, 3]
