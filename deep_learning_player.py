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

    def num_flat_features(self):
        return self.conv_to_linear_params_size

    def train_model(self):

        from data_handler import DataHandler

        learning_rate = 0.01
        momentum = 0.5
        # batch_size > 32 for all samples
        batch_size = 1000

        self.training_data = DataHandler.get_training_data(batch_size=batch_size)

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.train()

        for data in self.training_data:
            sample, target = Variable(FloatTensor([[data[0]]])), Variable(FloatTensor([data[1]]))

            optimizer.zero_grad()
            output = self(sample)

            criterion = nn.MSELoss()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        '''for index, training_game in enumerate(self.samples):

            # change batch size here
            for game_state in training_game:
                sample, target = Variable(FloatTensor([[game_state]])), Variable(FloatTensor(self.labels[index]))

                optimizer.zero_grad()
                output = self(sample)

                criterion = nn.MSELoss()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()'''

    def train_epoch(self, optimizer, epoch):
        pass


'''print "Test DeepLearningPlayer"
player = DeepLearningPlayer(color=1, time_limit=5, headless=True)
player.model.train_model()'''
