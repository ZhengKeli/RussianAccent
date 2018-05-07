from contexts.dataContext import *
from contexts.graphContext import *


def train_graph(train_count=1000):
	return graphTools.train_graph(session(), train_words_data, train_accents_data, train_count)


def test_graph(test_count=200):
	return graphTools.test_graph(session(), test_words_data, test_accents_data, test_count)
