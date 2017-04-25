import random
import h5py
import os
import torch

class DataHandler:

    WIN = 1
    LOSS = 0

    def __init__(self):
        self.load_training_data()

    @classmethod
    def __load_training_data__(cls):

        hdf = h5py.File("./TrainingData/samples.hdf5", "a")

        cls.games_won = [game.values() for game in hdf["win"].values()]
        cls.games_lost = [game.values() for game in hdf["loss"].values()]

        #print "Successfully loaded %i games" % (len(cls.games_won) + len(cls.games_lost))

    @classmethod
    def get_training_data(cls, batch_size=1000, shuffle=False):
        """get a list of training samples with their corresponding training labels in the following structure:
        [[training_sample 8x8, training_label], [training_sample 8x8, training_label], ..]
        :batch_size: the number of board states to be randomly chosen from each game"""

        # --------------------------------------------------------
        games = zip(cls.games_won, cls.games_lost)

        training_data = []
        for game_won, game_lost in games:
            training_data.extend([(sample.value, cls.WIN) for sample in random.sample(game_won, min(batch_size, len(game_won)))])
            training_data.extend([(sample.value, cls.LOSS) for sample in random.sample(game_lost, min(batch_size, len(game_lost)))])

        # --------------------------------------------------------

        if shuffle:
            random.shuffle(training_data)

        #print "successfully loaded %i training samples" % len(training_data)
        return training_data

    @classmethod
    def store_weights(cls, player_name, model):
        if not os.path.exists("./Weights"):
            os.makedirs("./Weights")

        torch.save(model, "./Weights/%s.pth" % player_name)

    @classmethod
    def load_weights(cls, player_name):
        model = torch.load("./Weights/%s.pth" % player_name)
        return model


def format_time(seconds):
    m,s = divmod(seconds, 60)
    h,m = divmod(m, 60)
    total_time = ""
    if h>0:
        total_time += "%ih " % h
    if m>0:
        total_time += "%im " % m
    total_time += "%is" % s

    return total_time


DataHandler.__load_training_data__()
