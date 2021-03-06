from player import Player
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import FloatTensor
from copy import deepcopy
from data_handler import DataHandler
import time
import numpy
import math

# WARNING: pyTorch only supports mini batches!
# see http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html for details

#################################################################################
#  Only replace the deep player import in othello.py with the following:        #
#  from deep_learning_player2 import DeepLearningPlayer2 as DeepLearningPlayer  #
#################################################################################

class DeepLearningPlayer2(Player):

    name = "DeepLearningPlayer2"

    def __init__(self, color="black", time_limit=5, gui=None, headless=False, epochs=5, batch_size=100):
        super(DeepLearningPlayer2, self).__init__(color, time_limit, gui, headless)
        self.model = Net()

        if torch.cuda.is_available():
            self.model.cuda()
            print "CUDA activated"

        #print(self.model)
        #need to change the extend data handler for this model too
        '''
        try:
            self.model = DataHandler.load_weights(self.name)
        except Exception:
           self.train_model(epochs=epochs, batch_size=batch_size)
        '''
        

    def train_model(self, epochs, batch_size):
        self.model.train_model(epochs=epochs, batch_size=batch_size)
        DataHandler.store_weights(player_name=self.name, model=self.model)

    def get_move(self):
        #moves = self.current_board.get_valid_moves(self.color)

        # Find the best next move
        move = self.__predict_move__()

        print "Next move: " , move

        self.apply_move(move)
        return self.current_board

    def __predict_move__(self):
        board = deepcopy(self.current_board)

        sample = FloatTensor([[board.get_representation(self.color)]])
        if torch.cuda.is_available():
            sample = sample.cuda()
        
        sample = Variable(sample)
        self.model(sample)
        return self.model.get_next_move(sample)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_to_linear_params_size = 64*8*8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=20, out_channels=24, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=24, out_channels=28, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=28, out_channels=32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=self.conv_to_linear_params_size, out_features=self.conv_to_linear_params_size/2)  # Channels x Board size (was 4x4 for some reason)
        self.fc2 = nn.Linear(in_features=self.conv_to_linear_params_size/2, out_features=self.conv_to_linear_params_size/4)
        self.fc3 = nn.Linear(in_features=self.conv_to_linear_params_size/4, out_features=self.conv_to_linear_params_size/16)
        self.fc4 = nn.Linear(in_features=self.conv_to_linear_params_size/16, out_features=self.conv_to_linear_params_size/32)
        self.fc5 = nn.Linear(in_features=self.conv_to_linear_params_size/32, out_features=self.conv_to_linear_params_size/64)
        self.fc6 = nn.Linear(in_features=self.conv_to_linear_params_size/64, out_features=self.conv_to_linear_params_size/64)
        self.fc7 = nn.Linear(in_features=self.conv_to_linear_params_size/64, out_features=self.conv_to_linear_params_size/64)
        self.fc8 = nn.Linear(in_features=self.conv_to_linear_params_size/64, out_features=self.conv_to_linear_params_size/64)
        self.fc9 = nn.Linear(in_features=self.conv_to_linear_params_size/64, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.view(-1, self.num_flat_features())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x

    def get_next_move(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.view(-1, self.num_flat_features())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc8(x))

        #Find the index with maximum value
        max_val, max_idx = torch.topk(x, 1)
        #Convert it to double
        max_idx = max_idx.data.numpy()[0]
        #Compute row and column for the next move from a 64 dimensional tensor
        row = math.floor(max_idx / 8)
        col = max_idx[0] - (row * 8)

        return (int(row),int(col))

    def num_flat_features(self):
        return self.conv_to_linear_params_size

    def train_model(self, epochs=1, batch_size=100):
        print "training Model"

        learning_rate = 0.01
        momentum = 0.5
        start_time = time.time()

        if not self.optimizer:
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.train()

        for i in range(epochs):
            epoch_time = time.time()
            self.train_epoch(optimizer=self.optimizer, batch_size=batch_size)
            print "Successively trained %s epochs (epoch timer: %s)" % (i+1, DataHandler.format_time(time.time()-epoch_time))

        total_time = DataHandler.format_time(time.time()-start_time)

        print "Finished training of %i epochs in %s" % (epochs, total_time)

    def train_epoch(self, optimizer, batch_size):

        training_data = DataHandler.get_training_data(batch_size=batch_size)

        for data in training_data:
            sample, target = FloatTensor([[data[0]]]), FloatTensor([data[1]])
            if torch.cuda.is_available():
                sample, target = sample.cuda(), target.cuda()
            sample, target = Variable(sample), Variable(target)
            optimizer.zero_grad()
            output = self(sample)

            #Negative Log-Likelihood loss
            criterion = nn.MSELoss()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


# print "Test DeepLearningPlayer"
# player = DeepLearningPlayer(color=1, time_limit=5, headless=True, epochs=2)
'''from board import Board
board = Board()
player.set_current_board(board)
move = player.get_move()
print "DeepLearningPlayer's move: "
print move.get_representation(1)'''
#DeepLearningPlayer.train_model(epochs=20, batch_size=100)
