from config import BLACK, WHITE
import copy

class Logger:

    # list of boards
    player_one_moves = []
    player_two_moves = []

    @classmethod
    def report(cls, color, original_board):
        board = copy.deepcopy(original_board)

        # player one, same as in othello.py
        if color == BLACK:
            cls.player_one_moves.append(board)
        elif color == WHITE:
            cls.player_two_moves.append(board)

        print('Player %s logged his move' % color)
        print('Number of moves logged: %i' % len(cls.player_one_moves))

    @classmethod
    def report_winner(cls, winner_color):

        cls.write_to_file()

        cls.player_one_moves = []
        cls.player_two_moves = []

        print('Game Over')
        print('Player %s won' % winner_color)
        print('Number of Moves: %i' % len(cls.player_one_moves))

    def write_to_file(self):
        # TODO: Implement this
        pass
