from data_handler import DataHandler
from board import Board
from deep_learning_player import DeepLearningPlayer
from config import WHITE, BLACK

from torch import FloatTensor
from torch.autograd import Variable

player = DeepLearningPlayer(color=WHITE, time_limit=1, headless=True, epochs=1, batch_size=100)
test_data = DataHandler.get_test_data()

score = 0.0
for sample in test_data:
    prediction = player.evaluate_board(Board(sample[0]), WHITE, BLACK)
    prediction = prediction.data.numpy()[0][0]

    if (prediction > 0.5) == (sample[1] == 1):
        score += 1

    print "predicting: %s, score: %s" % (prediction, score)


print "score: %s" % score
print "Accuracy: %s over %s test samples" % (score/len(test_data), len(test_data))
