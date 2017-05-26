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
from game_ai import GameArtificialIntelligence

# WARNING: pyTorch only supports mini batches!
# see http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html for details


class DeepLearningPlayer(Player):

    name = "DeepLearningPlayer"

    def __init__(self, color="black", time_limit=5, gui=None, headless=False, epochs=5, batch_size=100):
        super(DeepLearningPlayer, self).__init__(color, time_limit, gui, headless)
        self.model = Net()
        self.ai = GameArtificialIntelligence(self.evaluate_board);

        if torch.cuda.is_available():
            self.model.cuda(0)
            print "CUDA activated"

        # print(self.model)

        try:
            self.model = DataHandler.load_weights(self.name)
        except Exception:
            self.train_model(epochs=epochs, batch_size=batch_size)

    def train_model(self, epochs=10, batch_size=100, continue_training=False):
        losses = self.model.train_model(epochs=epochs, batch_size=batch_size, continue_training=continue_training)
        DataHandler.store_weights(player_name=self.name, model=self.model)
        return losses

    def train_model_on_curriculum(self, epochs_per_stage=1, final_epoch=30, continue_training=False):
        final_epoch = min(final_epoch, 30)
        losses = self.model.train_model_on_curriculum(epochs_per_stage=epochs_per_stage, final_epoch=final_epoch, continue_training=continue_training)
        DataHandler.store_weights(player_name=self.name + "_curriculum", model=self.model)
        return losses

    def evaluate_board(self, board, color, other_player):
        sample = FloatTensor([[board.get_representation(color)]])

        if torch.cuda.is_available():
            sample = sample.cuda(0)
        sample = Variable(sample)

        return self.model(sample)

    def get_move(self):
        # return self.get_move_alpha_beta()
        moves = self.current_board.get_valid_moves(self.color)

        # predict value for each possible move
        predictions = [(self.__predict_move__(move), move) for move in moves]

        # print "Chose move with prediction [%s]" % max(predictions)[0]
        self.apply_move(max(predictions)[1])
        return self.current_board

    def get_move_alpha_beta(self):
        move = self.ai.move_search(self.current_board, self.time_limit, self.color, (self.color % 2) + 1)
        self.apply_move(move)
        return self.current_board

    def __predict_move__(self, move):
        board = deepcopy(self.current_board)
        board.apply_move(move, self.color)

        sample = FloatTensor([[board.get_representation(self.color)]])
        if torch.cuda.is_available():
            sample = sample.cuda(0)
        sample = Variable(sample)

        return self.model(sample)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # '''
        self.conv_to_linear_params_size = 16*8*8
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=self.conv_to_linear_params_size,    out_features=self.conv_to_linear_params_size/ 4)  # Channels x Board size (was 4x4 for some reason)
        self.fc2 = nn.Linear(in_features=self.conv_to_linear_params_size/ 4, out_features=self.conv_to_linear_params_size/ 16)
        self.fc3 = nn.Linear(in_features=self.conv_to_linear_params_size/ 16, out_features=1)
        # '''

        '''
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        '''

        self.learning_rate = 0.01
        self.criterion = torch.nn.MSELoss(size_average=False)
        # self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=True)

    def forward(self, x):
        # '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(-1, self.num_flat_features())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # '''

        '''
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        '''

        return x

    def num_flat_features(self):
        return self.conv_to_linear_params_size

    def train_model(self, epochs=1, batch_size=100, continue_training=False):
        print "training Model"

        try:
            if continue_training:
                self.optimizer
            else:
                self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        except AttributeError:
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        self.train()
        losses = []
        start_time = time.time()
        for i in range(epochs):
            training_data = DataHandler.get_training_data(batch_size=batch_size)
            losses.extend(self.train_epoch(optimizer=self.optimizer, training_data=training_data, epoch_id=i))

        total_time = DataHandler.format_time(time.time() - start_time)
        print "Finished training of %i epochs in %s" % (epochs+1, total_time)
        return losses

    def train_model_on_curriculum(self, epochs_per_stage, final_epoch, continue_training=False):

        try:
            if continue_training:
                self.optimizer
            else:
                self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        except AttributeError:
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        self.train()
        losses = []
        start_time = time.time()
        for epoch in range(final_epoch*epochs_per_stage):
            # training_data = get_dummy_training_data((16000 * 60 / 10)) # 10% of training set size
            training_data = DataHandler.get_curriculum_training_data(epoch/epochs_per_stage)
            losses.extend(self.train_epoch(optimizer=self.optimizer, training_data=training_data, epoch_id=epoch))

        total_time = DataHandler.format_time(time.time() - start_time)
        print "Finished training of %i epochs in %s" % (epoch+1, total_time)
        return losses

    def train_epoch(self, optimizer, training_data, epoch_id='unknown'):
        epoch_time = time.time()

        accumulated_loss = 0
        average_losses = []
        training_data_length = len(training_data)
        percent_done = 0
        for index, data in enumerate(training_data):
            sample, target = FloatTensor([[data[0]]]), FloatTensor([data[1]])
            if torch.cuda.is_available():
                sample, target = sample.cuda(0), target.cuda(0)
            sample, target = Variable(sample), Variable(target)

            optimizer.zero_grad()
            output = self(sample)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            accumulated_loss += loss.data[0]

            if percent_done - 100 * index // training_data_length != 0:
                percent_done = 100 * index // training_data_length
                average_losses.append(accumulated_loss/(index+1))
                print('Finished %s%% of epoch %s | average loss: %s' % (percent_done, epoch_id+1, accumulated_loss/(index+1)))

        print "Successively trained %s epochs (epoch timer: %s)" % (epoch_id+1, DataHandler.format_time(time.time() - epoch_time))
        return average_losses


import numpy
def get_dummy_training_data(sample_size):
        return [(numpy.array([[i%2]*8]*8), i%2) for i in range(sample_size)]
