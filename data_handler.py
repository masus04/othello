import random
import h5py
import os
import torch


class DataHandler:

    WIN = 1
    LOSS = 0

    def __init__(self):
        print "Initializing DataHandler"
        self.__load_training_data__()

    @classmethod
    def __load_training_data__(self):

        hdf = h5py.File("./TrainingData/samples.hdf5", "a")

        self.games_won = [game.values() for game in hdf["win"].values()]
        self.games_lost = [game.values() for game in hdf["loss"].values()]

        print "Successfully loaded %i games" % (len(self.games_won) + len(self.games_lost))

    @classmethod
    def get_training_data(cls, batch_size=100, shuffle=False):
        """get a list of training samples with their corresponding training labels in the following structure:
        [[training_sample 8x8, training_label], [training_sample 8x8, training_label], ..]
        :batch_size: the number of board states to be randomly chosen from each game"""

        # --------------------------------------------------------
        try:
            cls.games_won and cls.games_lost
        except Exception:
            cls.__load_training_data__()

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
    def get_test_data(self):
        hdf = h5py.File("./TrainingData/test.hdf5", "a")

        self.games_won = [game.values() for game in hdf["win"].values()]
        self.games_lost = [game.values() for game in hdf["loss"].values()]

        test_data = [(item.value, 1) for sublist in self.games_won for item in sublist]
        test_data.extend([(item.value, 0) for sublist in self.games_lost for item in sublist])
        return test_data

    @classmethod
    def store_weights(cls, player_name, model):
        if not os.path.exists("./Weights"):
            os.makedirs("./Weights")

        torch.save(model, "./Weights/%s.pth" % player_name)

    @classmethod
    def load_weights(cls, player_name):
        model = torch.load("./Weights/%s.pth" % player_name)
        return model

    @classmethod
    def format_time(cls, seconds):
        m,s = divmod(seconds, 60)
        h,m = divmod(m, 24)
        total_time = ""
        if h>0:
            total_time += "%ih " % h
        if m>0:
            total_time += "%im " % m
        total_time += "%is" % s

        return total_time

    @classmethod
    def merge_samples(cls):
        merged_file = h5py.File("./TrainingData/samples.hdf5", "a")

        sample_files = os.listdir("./TrainingData")
        for file_name in sample_files:
            sample_file = h5py.File("./TrainingData/" + file_name)

            for game in sample_file["win"].values():
                if not game.name in merged_file["win"]:
                    game.copy(source=game, dest=merged_file["win"], name=game.name)

            for game in sample_file["loss"].values():
                if not game.name in merged_file["loss"]:
                    game.copy(source=game, dest=merged_file["loss"], name=game.name)

# DataHandler.merge_samples()
# DataHandler.get_test_data()
DataHandler()
