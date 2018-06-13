import tensorflow as tf

import conf
from tools import graphTools

sess = tf.Session()


def get_sess():
	return sess


def create_graph():
	return graphTools.create_graph(sess)


def log_graph_shape():
	graphTools.log_graph_shape(sess, conf.get_log_dir(conf.default_version))


def save_graph(step=None):
	if step is None:
		step = sess.run("train/training_step:0", {})
	graphTools.save_graph(sess, conf.get_graph_file(conf.default_version, step))
	print("graph saved (step = ", step, ")")


def load_graph(step=None):
	if step is None:
		step = conf.get_latest_step(conf.default_version)
	meta_file = conf.get_graph_meta_file(conf.default_version, step)
	graph_dir = conf.get_graph_dir(conf.default_version, step)
	return graphTools.load_graph(sess, meta_file, graph_dir)


def use_graph(words_data):
	return graphTools.use_graph(sess, words_data)
