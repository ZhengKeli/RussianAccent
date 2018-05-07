import tensorflow as tf


def session():
	if tf.get_default_session() is None:
		raise BaseException("graph not loaded!")
	return tf.get_default_session()


def get_or_create_session():
	if tf.get_default_session() is None:
		tf.InteractiveSession()
	return tf.get_default_session()
