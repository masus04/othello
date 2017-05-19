import time
from deep_learning_player import DeepLearningPlayer
from data_handler import DataHandler

print "Training Model"
player = DeepLearningPlayer(color=1, time_limit=5, headless=True, epochs=2)

start_time = time.time()
player.train_model(epochs=1, batch_size=100)
print "Training successfull, took %s" % DataHandler.format_time(time.time() - start_time)
