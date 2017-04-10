from player import Player
import numpy as np
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

        self.apply_move(moves[0])
        return self.current_board

    def init_network(self):
        model = Net()
        print(model)

        model.train_model()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 8, 3)
        self.conv3 = nn.Conv2d(8, 12, 3)
        self.conv4 = nn.Conv2d(12, 12, 3)
        self.conv5 = nn.Conv2d(12, 12, 3)
        self.conv6 = nn.Conv2d(12, 12, 3)
        self.conv7 = nn.Conv2d(16, 16, 3)
        self.conv8 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(16*4*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        self.training_data = self.load_training_data()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def train_model(self):
        learning_rate = 0.01
        momentum = 0.5
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.train()
        for training_game in self.training_data:
            game_states = training_game[0]
            label = training_game[1]
            print "sample_len: %i, label: %i" % (len(game_states), label)

    def train_epoch(self, optimizer, epoch):
        pass

    @classmethod
    def load_training_data(cls):
        import h5py

        hdf = h5py.File("./TrainingData/samples.hdf5", "a")

        samples = cls.retrieve_training_data(hdf["win"])
        labels = ([1] * len(samples))

        samples.extend(cls.retrieve_training_data(hdf["loss"]))
        labels.extend([0] * (len(samples) - len(labels)))

        training_data = zip(samples, labels)

        print "Successfully loaded %i training samples" % len(training_data)
        return training_data

    @classmethod
    def retrieve_training_data(cls, group):
        arr = []
        for game_group in group.values():
            arr.append([game_state.value for game_state in game_group.values()])

        return arr
