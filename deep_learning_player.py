from player import Player
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# WARNING: pyTorch only supports mini batches!
# see http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html for details


class DeepLearningPlayer(Player):

    def __init__(self, color="black", time_limit=5, gui=None, headless=False):
        super(DeepLearningPlayer, self).__init__(color, time_limit, gui, headless)
        self.init_network()

    def get_move(self):
        moves = self.current_board.get_valid_moves(self.color)

        # predict value for each possible move
        for move in moves:
            pass

    def init_network(self):
        learning_rate = 0.01
        momentum = 0.5
        model = Net()
        print(model)

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        model.train_model(optimizer)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 4)
        self.conv3 = nn.Conv2d(8, 16, 8)
        self.fc1 = nn.Linear(16*4*4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self):
        x = F.relu(self.conv1)
        x = F.relu(self.conv2)
        x = F.relu(self.conv3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def train_model(self, optimizer):
        pass

    def train_epoch(self, optimizer, epoch):
        pass


# test network
net = Net()
print(net)
