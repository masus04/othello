import time
from deep_learning_player import DeepLearningPlayer
from data_handler import DataHandler
from test_network import test_network
import datetime
import matplotlib.pyplot as plt


def train_curriculum():
    print "Training Model"
    player = DeepLearningPlayer(color=1, time_limit=5, headless=True, epochs=0)

    start_time = time.time()
    accuracies = []
    while (accuracies == []):
        #while datetime.datetime.today().day % 2 != 0:
        #   time.sleep(3600)

        #losses = player.train_model(epochs=1, batch_size=10, continue_training=True)
        losses = player.train_model_on_curriculum(epochs_per_stage=5, final_epoch=1, continue_training=True)
        print "Training successfull, took %s" % DataHandler.format_time(time.time() - start_time)
        acc = test_network(player)
        print "Training Error / Accuracy: %s" % acc
        accuracies.append(acc)
        plt.plot(accuracies, 'r--')
        plt.plot(accuracies, 'g^')
        plt.plot(losses[99::100], 'b--')
        plt.ylabel('Training Error / Accuracy')
        plt.xlabel('Epochs')
        plt.savefig('acc.png')
        start_time = time.time()


def train_network():
    print "Training Model"
    player = DeepLearningPlayer(color=1, time_limit=5, headless=True, epochs=0)

    start_time = time.time()
    accuracies = []
    while (True):
        #while datetime.datetime.today().day % 2 != 0:
        #   time.sleep(3600)

        losses = player.train_model(epochs=1, batch_size=10, continue_training=True)
        #losses = player.train_model_on_curriculum(epochs_per_stage=5, final_epoch=1, continue_training=True)
        print "Training successfull, took %s" % DataHandler.format_time(time.time() - start_time)
        acc = test_network(player)
        print "Training Error / Accuracy: %s" % acc
        accuracies.append(acc)
        plt.plot(accuracies, 'r--')
        plt.plot(accuracies, 'g^')
        plt.plot(losses[99::100], 'b--')
        plt.ylabel('Training Error / Accuracy')
        plt.xlabel('Epochs')
        plt.savefig('acc.png')
        start_time = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data_handler import DataHandler
import time
import numpy
from torch import FloatTensor


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # '''
        self.conv_to_linear_params_size = 16*8*8
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels= 8, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels= 8, out_channels=12, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=self.conv_to_linear_params_size,    out_features=self.conv_to_linear_params_size/ 2)  # Channels x Board size (was 4x4 for some reason)
        self.fc2 = nn.Linear(in_features=self.conv_to_linear_params_size/ 2, out_features=self.conv_to_linear_params_size/ 4)
        self.fc3 = nn.Linear(in_features=self.conv_to_linear_params_size/ 4, out_features=1)
        # '''

        '''
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        '''

        self.learning_rate = 0.001
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(size_average=False)
        #self.criterion = nn.BCELoss(weight=None, size_average=True)

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

    def train_model(self, sample_size, epochs):
        print "training Model"

        self.train()
        training_data = get_dummy_training_data(sample_size)
        start_time = time.time()

        for i in range(epochs):
            accumulated_loss = 0
            epoch_timer = time.time()
            for index, data in enumerate(training_data):
                sample, target = FloatTensor([[data[0]]]), FloatTensor([data[1]])
                if torch.cuda.is_available():
                    sample, target = sample.cuda(0), target.cuda(0)
                sample, target = Variable(sample), Variable(target)

                self.optimizer.zero_grad()
                output = self(sample)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                accumulated_loss += loss.data[0]

            print "Finished epoch: %s in %s | average loss: %s" % (i+1, DataHandler.format_time(time.time() - epoch_timer), accumulated_loss/(index+1))
        print "Finished training in %s" % DataHandler.format_time(time.time() - start_time)


def get_dummy_training_data(sample_size):
        return [(numpy.array([[i%2]*8]*8), i%2) for i in range(sample_size)]


def train_dummy(sample_size, epochs):
    model = Net()
    model.train_model(sample_size=sample_size, epochs=epochs)

""" --- ! Choose training mode here ! --- """

DataHandler.merge_samples()

""" --! Training !-- """
# train_network()
train_curriculum()
# train_dummy(sample_size=16000*60 / 10, epochs=100)  # 16000 * 60 = size of our training set, so each episode is 10% of an epoch in real training
