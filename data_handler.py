import random
import h5py
import os
import torch
import numpy

class DataHandler:

    WIN = 1
    LOSS = 0

    def __init__(self):
        print "Initializing DataHandler"
        self.init_training_data()

    @classmethod
    def init_training_data(cls):
        try:
            hdf = h5py.File("./TrainingData/samples.hdf5", "a")
        except Exception:
            cls.merge_samples()

        try:
            cls.games_won and cls.games_lost and cls.games and cls.test_games_won and cls.test_games_lost
        except Exception:
            cls.games_won = [game.values() for game in hdf["win"].values()]
            cls.games_lost = [game.values() for game in hdf["loss"].values()]
            cls.games = zip(cls.games_won, cls.games_lost)

            hdf = h5py.File("./TrainingData/Test/test.hdf5", "a")
            cls.test = {}
            cls.test_games_won = [game.values() for game in hdf["win"].values()]
            cls.test_games_lost = [game.values() for game in hdf["loss"].values()]
            cls.test_games = zip(cls.test_games_won, cls.test_games_lost)

            print "Successfully loaded %i games" % (len(cls.games_won) + len(cls.games_lost))

    @classmethod
    def get_training_data(cls, batch_size=100, shuffle=False):
        """get a list of training samples with their corresponding training labels in the following structure:
        [[training_sample 8x8, training_label], [training_sample 8x8, training_label], ..]
        :batch_size: the number of board states to be randomly chosen from each game"""

        # --------------------------------------------------------
        cls.init_training_data()

        training_data = []
        for game_won, game_lost in cls.games:
            training_data.extend([(sample.value, cls.WIN) for sample in random.sample(game_won, min(batch_size, len(game_won)))])
            training_data.extend([(sample.value, cls.LOSS) for sample in random.sample(game_lost, min(batch_size, len(game_lost)))])

        # --------------------------------------------------------

        if shuffle:
            random.shuffle(training_data)

        #print "successfully loaded %i training samples" % len(training_data)
        return DataHandler.transform_to_positive(training_data)

    @classmethod
    def get_test_data(cls):

        cls.init_training_data()

        test_data = [(item.value, 1) for sublist in cls.test_games_won for item in sublist]
        test_data.extend([(item.value, 0) for sublist in cls.test_games_lost for item in sublist])

        return cls.transform_to_positive(test_data)

    @classmethod
    def get_curriculum_training_data(cls, iteration):

        cls.init_training_data()

        training_data = []
        for game_won, game_lost in cls.games:
            training_data.extend([(sample.value, cls.WIN) for sample in game_won if str(iteration) in sample.name])
            training_data.extend([(sample.value, cls.LOSS) for sample in game_lost if str(iteration) in sample.name])

        return cls.transform_to_positive(training_data)

    @classmethod
    def get_curriculum_test_data(cls, iteration):

        cls.init_training_data()

        test_data = []
        for game_won, game_lost in cls.test_games:
            test_data.extend([(sample.value, cls.WIN) for sample in game_won if str(iteration) in sample.name])
            test_data.extend([(sample.value, cls.LOSS) for sample in game_lost if str(iteration) in sample.name])

        return cls.transform_to_positive(test_data)

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
            file_name = "./TrainingData/" + file_name
            if os.path.isfile(file_name):
                sample_file = h5py.File(file_name, "r")

                if not 'win' in merged_file.keys():
                    merged_file.create_group("win")
                    merged_file.create_group("loss")

                for game in sample_file["win"].values():
                    if not game.name in merged_file["win"]:
                        game.copy(source=game, dest=merged_file["win"], name=game.name)

                for game in sample_file["loss"].values():
                    if not game.name in merged_file["loss"]:
                        game.copy(source=game, dest=merged_file["loss"], name=game.name)

    @classmethod
    def transform_to_positive(cls, training_data):
        return [(positive_board(sample[0]), sample[1]) for sample in training_data]

    @classmethod
    def generate_positive_training_data(cls):
        samples = h5py.File("./TrainingData/samples.hdf5", "r")
        positive_samples = h5py.File("./TrainingData/positive_samples.hdf5", "a")
        if not 'win' in positive_samples.keys():
            positive_samples.create_group("win")
            positive_samples.create_group("loss")

        for game in samples["win"].values():
            if not game.name in positive_samples["win"]:
                positive_samples.create_group(game.name)
                for state in game.values():
                    positive_samples.create_dataset(name=state.name, data=positive_board(state.value))

            if not game.name in positive_samples["loss"]:
                positive_samples.create_group(game.name)
                for state in game.values():
                    positive_samples.create_dataset(name=state.name, data=positive_board(state.value))


        samples.close()
        positive_samples.close()


def positive_board(board_state):
    from copy import deepcopy
    positive = deepcopy(board_state)

    for i in range(8):
        for j in range(8):
            if positive[i][j] < 0:
                positive[i][j] = 3

            elif positive[i][j] == 0:
                positive[i][j] = 2

    return positive

# DataHandler.generate_positive_training_data()
