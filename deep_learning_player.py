from player import Player
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import FloatTensor
from copy import deepcopy
from data_handler import DataHandler

# WARNING: pyTorch only supports mini batches!
# see http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html for details


class DeepLearningPlayer(Player):

    name = "DeepLearningPlayer"

    def __init__(self, color="black", time_limit=5, gui=None, headless=False, epochs=3, batch_size=1000):
        super(DeepLearningPlayer, self).__init__(color, time_limit, gui, headless)
        self.model = Net()

        if torch.cuda.is_available():
            self.model.cuda()

        print(self.model)

        try:
            self.model = DataHandler.load_weights(self.name)
        except Exception:
            self.model.train_model(epochs=epochs, batch_size=batch_size)
            DataHandler.store_weights(player_name=self.name, model=self.model)

    def get_move(self):
        moves = self.current_board.get_valid_moves(self.color)

        # predict value for each possible move
        predictions = [(self.__predict_move__(move), move) for move in moves]

        print "Chose move with prediction [%s]" % max(predictions)[0]
        self.apply_move(max(predictions)[1])
        return self.current_board

    def __predict_move__(self, move):
        board = deepcopy(self.current_board)
        board.apply_move(move, self.color)

        sample = FloatTensor([[board.get_representation(self.color)]])
        if torch.cuda.is_available():
            sample = sample.cuda()
        sample = Variable(sample)

        return self.model.forward(sample)


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

    def train_model(self, epochs=1, batch_size=1000):
        print "training Model"

        learning_rate = 0.01
        momentum = 0.5

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.train()

        for i in range(epochs):
            self.train_epoch(optimizer=self.optimizer, batch_size=batch_size)
            print "Successively trained %i epochs" % (i+1)

    def train_epoch(self, optimizer, batch_size):

        training_data = DataHandler.get_training_data(batch_size=batch_size)

        for data in training_data:
            sample, target = FloatTensor([[data[0]]]), FloatTensor([data[1]])
            if torch.cuda.is_available():
                sample, target = sample.cuda(), target.cuda()
            sample, target = Variable(sample), Variable(target)

            optimizer.zero_grad()
            output = self(sample)

            criterion = nn.MSELoss()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


'''print "Test DeepLearningPlayer"
player = DeepLearningPlayer(color=1, time_limit=5, headless=True, epochs=1, batch_size=1)
from board import Board
board = Board()
player.set_current_board(board)
move = player.get_move()
print "DeepLearningPlayer's move: "
print move.get_representation(1)'''
