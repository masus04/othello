from data_handler import DataHandler
from board import Board
from deep_learning_player import DeepLearningPlayer
from config import WHITE, BLACK

from torch import FloatTensor
from torch.autograd import Variable

def test_network(player, test_data):

    score = 0.0
    for index, sample in enumerate(test_data):
        prediction = player.evaluate_board(Board(sample[0]), WHITE, BLACK)
        prediction = prediction.data.numpy()[0][0]

        if (prediction > 0.5) == (sample[1] == 1):
            score += 1

        if float(index)/len(test_data) % 1 == 0:
            print "Evaluating network: %s%% done" % str(index/len(test_data))
            # print "predicting: %s, score: %s" % (prediction, score)


    # print "score: %s" % score
    # print "Accuracy: %s over %s test samples" % (score/len(test_data), len(test_data))
    
    return score/len(test_data)

player = DeepLearningPlayer(color=WHITE, time_limit=1, headless=True, epochs=0, batch_size=100)
print "Evaluation error: %s" % str(test_network(player, DataHandler.get_curriculum_test_data(0)))
