import conf
from contexts.sessionContext import *
from tools import graphTools


def create_graph():
	return graphTools.create_graph(get_or_create_session())


def save_graph():
	return graphTools.save_graph(session(), conf.graph_file)


def log_graph_shape():
	tf.summary.FileWriter(conf.log_dir).add_graph(session().graph)


def load_graph():
	return graphTools.load_graph(get_or_create_session(), conf.graph_meta_file, conf.graph_dir)


def use_graph(words_data):
	return graphTools.use_graph(session(), words_data)
