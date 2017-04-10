#!/usr/bin/env python

import board
import player
from deep_learning_player import DeepLearningPlayer
import numpy as np
from gui import Gui
from config import BLACK, WHITE, HUMAN, COMPUTER
from game_state_logger import Logger
from heuristic import OthelloHeuristic
from random import randint
import properties

class Othello:

    # Default values, they are overridden by values in the properties file
    headless = True
    number_of_games = 100
    timeout = 5;

    def __init__(self):
        if hasattr(properties, "timeout"):
            self.timeout = properties.timeout
        if hasattr(properties, "number_of_games"):
            self.number_of_games = properties.number_of_games

        if self.headless:
            self.setup_headless_game()
        else:
            self.gui = Gui()
            self.setup_game()
        self.setup_headless_game()

    def setup_headless_game(self):
        self.headless = True
        # player one, same as in game_state_logger.py
        self.now_playing = player.ComputerPlayer(color=BLACK, time_limit=self.timeout, headless=self.headless, strategy=randint(0,2))
        # player two, same as in game_state_logger.py
        self.other_player = DeepLearningPlayer(color=WHITE, time_limit=self.timeout, headless=self.headless)
        self.board = board.Board()
        Logger.set_player_names([self.now_playing.name, self.other_player.name])


    def setup_game(self):
        options = self.gui.show_options()
        if options['player_1'] == COMPUTER:
            self.now_playing = player.ComputerPlayer(BLACK, int(options['player_1_time']), self.gui)
        else:
            self.now_playing = player.HumanPlayer(BLACK, gui=self.gui)
        if options['player_2'] == COMPUTER:
            self.other_player = player.ComputerPlayer(WHITE, int(options['player_2_time']), self.gui)
        else:
            self.other_player = player.HumanPlayer(WHITE, gui=self.gui)
        if options.has_key('load_file'):
            self.board = board.Board(self.read_board_file(options['load_file']))
        else:
            self.board = board.Board()

    def read_board_file(self, file_name):
        f = open(file_name)
        lines = [line.strip() for line in f]
        f.close()
        board = np.zeros((8, 8), dtype=np.integer)
        # Read In Board File
        i = 0
        for line in lines[:8]:
            j = 0
            for char in line.split():
                board[i][j] = int(char)
                j += 1
            i += 1
        # Set Current Turn
        if int(lines[8]) == WHITE:
            self.now_playing, self.other_player = self.other_player, self.now_playing

        return board

    def run(self, games=1):
        print "Game started: %s vs %s, time limit: %is" % (self.now_playing.name, self.other_player.name, self.timeout)
        if not self.headless:
            self.gui.show_game(self.board)
        while True:
            winner = self.board.game_won()
            if winner is not None:
                Logger.report_winner(winner)
                break
            self.now_playing.set_current_board(self.board)
            if self.board.get_valid_moves(self.now_playing.color) != []:
                self.board = self.now_playing.get_move()
            if not self.headless:
                self.gui.update(self.board, self.other_player)
            self.now_playing, self.other_player = self.other_player, self.now_playing
        if not self.headless:
            self.gui.show_winner(winner, self.board)
        self.restart(games - 1)

    def restart(self, games):
        if games > 0:
            if self.headless:
                self.setup_headless_game()
            else:
                self.setup_game()
            self.run(games)

def main():
    game = Othello()
    game.run(Othello.number_of_games)

if __name__ == '__main__':
    main()
