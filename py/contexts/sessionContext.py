import tensorflow as tf

import conf
from tools import graphTools

sess = tf.Session()


def get_sess():
	return sess


def create_graph():
	return graphTools.create_graph(sess)


def save_graph():
	return graphTools.save_graph(sess, conf.graph_file)


def log_graph_shape():
	tf.summary.FileWriter(conf.log_dir).add_graph(sess.graph)


def load_graph():
	return graphTools.load_graph(sess, conf.graph_meta_file, conf.graph_dir)


def use_graph(words_data):
	return graphTools.use_graph(sess, words_data)
