import copy
import h5py
import os, sys
import numpy as np
from properties import uid

# Use first argument as uid
try:
    uid = sys.argv[1]
except TypeError:
    pass


class Logger:
    """
    Generates a hdf5 file and adds every game state of every played game to it
    Structure is as follows:

    samples (file name)
        win (group name)
            PlayerName_PlayerName_win_index (dataset name)
        loss
            PlayerName_PlayerName_loss_index (dataset name)

    The win/ loss groups encode the label. These contain all game states any player has lost/ won.

    """

    # player_moves[player][move]
    player_moves = [[], []]
    player_names = None
    min_depth = 100
    depth_sum = 0

    @classmethod
    def report(cls, color, original_board):
        board = copy.deepcopy(original_board)

        cls.player_moves[color-1].append(board)

    @classmethod
    def report_winner(cls, winner_color):
        looser_color = (winner_color % 2) + 1
        game_name = "%s_%s" % (cls.player_names[winner_color-1], cls.player_names[looser_color-1])

        hdf = cls.init_hdf5()

        # Find next available index
        i = 0
        while i>=0 :
            group_name = "uid:_%s_%s_win_%s_min_depth:%i" % (uid, game_name, format(i, "05d"), cls.min_depth)
            if group_name in hdf['win']:
                i += 1
            else:
                game_group = hdf["win"].create_group(group_name)
                i = 0

                for move in cls.player_moves[winner_color -1]:
                    game_group.create_dataset("game_state_%s" % format(i, "02d"), data=np.array(move.get_representation(winner_color)))
                    i += 1

                i = -1 # break condition

        # Find next available index
        i = 0
        while i>=0 :
            group_name = "uid:_%s_%s_loss_%s_min_depth:%i" % (uid, game_name, format(i, "05d"), cls.min_depth)
            if group_name in hdf['loss']:
                i += 1
            else:
                game_group = hdf["loss"].create_group(group_name)
                i = 0

                for move in cls.player_moves[winner_color -1]:
                    game_group.create_dataset("game_state_%s" % format(i, "02d"), data=np.array(move.get_representation(winner_color)))
                    i += 1

                i = -1 # break condition

        print "-- | Player %s won | --" % winner_color
        print "Average depth: %i | Min depth: %i" % (cls.depth_sum / len(cls.player_moves[winner_color - 1]), cls.min_depth)

        cls.player_moves = [[], []]
        cls.depth_sum = 0

    @classmethod
    def report_depth(cls, depth):
        cls.depth_sum += depth
        cls.min_depth = min(cls.min_depth, depth)


    @classmethod
    def init_hdf5(cls):

        if not os.path.exists("./TrainingData"):
            os.makedirs("./TrainingData")

        # Open / create hdf5 file
        hdf = h5py.File("./TrainingData/samples.hdf5", "a")

        # Initialize groups (folders)
        if not 'win' in hdf.keys():
            hdf.create_group("win")
            hdf.create_group("loss")

        return hdf

    @classmethod
    def set_player_names(cls, name_list):
        cls.player_names = name_list
