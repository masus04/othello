import time
from deep_learning_player import DeepLearningPlayer
from data_handler import DataHandler
from test_network import test_network
import datetime
import matplotlib.pyplot as plt

print "Training Model"
player = DeepLearningPlayer(color=1, time_limit=5, headless=True, epochs=0)

start_time = time.time()
accuracies = []
while (True):
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
