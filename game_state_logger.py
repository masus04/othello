import copy
import h5py
import os
import numpy as np
from properties import uid


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
        while "uid:_%s_%s_win_%i" % (uid, game_name, i) in hdf['win']:
            i += 1
        for move in cls.player_moves[winner_color - 1]:
            hdf["win"].create_dataset("uid:_%s_%s_win_%i" % (uid, game_name, i), data=np.array(move.get_representation(winner_color)))
            i += 1

        # Find next available index
        i = 0
        while "uid:_%s_%s_loss_%i" % (uid, game_name, i) in hdf['loss']:
            i += 1
        for move in cls.player_moves[looser_color - 1]:
            hdf["loss"].create_dataset("uid:_%s_%s_loss_%i" % (uid, game_name, i), data=np.array(move.get_representation(looser_color)))
            i += 1

        print('-- | Player %s won | --' % winner_color)

        cls.player_moves = [[], []]

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
