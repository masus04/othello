from player import Player
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import FloatTensor

# WARNING: pyTorch only supports mini batches!
# see http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html for details


class DeepLearningPlayer(Player):

    def __init__(self, color="black", time_limit=5, gui=None, headless=False):
        super(DeepLearningPlayer, self).__init__(color, time_limit, gui, headless)

        self.model = Net()
        print(self.model)

        # model.train_model()

    def get_move(self):
        moves = self.current_board.get_valid_moves(self.color)

        # predict value for each possible move
        for move in moves:
            pass

        self.apply_move(moves[0])
        return self.current_board


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_to_linear_params_size = 16 * 8 * 8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=self.conv_to_linear_params_size, out_features=128)  # Channels x Board size (was 4x4 for some reason)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=1)

        """self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)  # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)"""

        self.load_training_data()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.num_flat_features())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

        """x = F.relu(self.conv1(x))  # Max pooling over a (2, 2) window
        x = F.relu(self.conv2(x))  # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x"""

    def num_flat_features(self):
        return self.conv_to_linear_params_size

    def train_model(self):
        learning_rate = 0.01
        momentum = 0.5
        batch_size = 5
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.train()
        for index, training_game in enumerate(self.samples):

            # change batch size here
            for game_state in training_game:
                sample, target = Variable(FloatTensor([[game_state]])), Variable(FloatTensor(self.labels[index]))

                optimizer.zero_grad()
                output = self(sample)

                criterion = nn.MSELoss()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def train_epoch(self, optimizer, epoch):
        pass

    @classmethod
    def load_training_data(cls):
        import h5py

        hdf = h5py.File("./TrainingData/samples.hdf5", "a")

        cls.samples = cls.retrieve_training_data(hdf["win"])
        cls.labels = [[1]] * len(cls.samples)

        cls.samples.extend(cls.retrieve_training_data(hdf["loss"]))
        cls.labels.extend([[1]] * (len(cls.samples) - len(cls.labels)))

        print "Successfully loaded %i training samples" % len(cls.labels)

    @classmethod
    def retrieve_training_data(cls, group):
        arr = []
        for game_group in group.values():
            arr.append([game_state.value for game_state in game_group.values()])

        return arr

print "Test DeepLearningPlayer"
player = DeepLearningPlayer(color=1, time_limit=5, headless=True)
player.model.train_model()
