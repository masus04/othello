from deep_learning_player import DeepLearningPlayer

print "Test DeepLearningPlayer"
player = DeepLearningPlayer(color=1, time_limit=5, headless=True, epochs=2)
player.train_model(epochs=20, batch_size=100)
