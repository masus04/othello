import random

class DataHandler:

    WIN = 1
    LOSS = 0

    def __init__(self):
        self.load_training_data()

    @classmethod
    def __load_training_data__(cls):
        import h5py

        hdf = h5py.File("./TrainingData/samples.hdf5", "a")

        cls.games_won = [game.values() for game in hdf["win"].values()]
        cls.games_lost = [game.values() for game in hdf["loss"].values()]

        print "Successfully loaded %i games" % (len(cls.games_won) + len(cls.games_lost))

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

        print "successfully loaded %i training samples" % len(training_data)
        return training_data


DataHandler.__load_training_data__()
data = DataHandler.get_training_data(1)
data
