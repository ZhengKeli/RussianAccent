import os

import numpy as np
import tensorflow as tf

import conf


def create_graph(sess):
	# arguments
	word_length = conf.max_word_length
	letter_count = len(conf.letters)
	letter_vector_size = 8
	
	rnn_memory_size = 32
	rnn_input_size = letter_vector_size
	rnn_output_size = rnn_memory_size
	
	rnn_cell_input_size = rnn_memory_size + rnn_input_size
	rnn_cell_output_size = rnn_output_size
	
	# constructions
	with tf.name_scope("input"):
		word = tf.placeholder(tf.int32, [None, word_length], name="word")
		accent = tf.placeholder(tf.int32, [None], name="accent")
		batch_count = tf.shape(word)[0]
		
		letter_vector_map = tf.Variable(tf.random_normal([letter_count, letter_vector_size], 0.4, 0.4), name="letter_vector_map")
		letters = tf.unstack(word, word_length, -1)
		letters_vector = [tf.gather(letter_vector_map, letter) for letter in letters]
		
		accent_sparse_index = tf.stack([tf.range(batch_count), accent], -1)
		accent_onehot = tf.sparse_to_dense(accent_sparse_index, [batch_count, word_length], 1.0)
	
	with tf.name_scope("forward_rnn"):
		forward_rnn_inputs = letters_vector
		forward_rnn_base_memory = tf.zeros([batch_count, rnn_memory_size], name="forward_rnn_base_memory")
		forward_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_cell_output_size, tf.nn.relu, reuse=tf.AUTO_REUSE)
		forward_rnn_outputs, _ = tf.nn.static_rnn(forward_rnn_cell, forward_rnn_inputs, forward_rnn_base_memory)
	with tf.name_scope("backward_rnn"):
		backward_rnn_inputs = letters_vector[::-1]
		backward_rnn_base_memory = tf.zeros([batch_count, rnn_memory_size], name="backward_rnn_base_memory")
		backward_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_cell_output_size, tf.nn.relu, reuse=tf.AUTO_REUSE)
		backward_rnn_outputs, _ = tf.nn.static_rnn(backward_rnn_cell, backward_rnn_inputs, backward_rnn_base_memory)
	
	with tf.name_scope("analyzer"):
		analyzer_cell = tf.layers.Dense(32, tf.nn.relu)
		analyzer_inputs = [tf.multiply(forward_rnn_outputs[cycle_id], backward_rnn_outputs[cycle_id]) for cycle_id in range(word_length)]
		analyzer_outputs = [analyzer_cell(analyzer_input) for analyzer_input in analyzer_inputs]
		analyzer_outputs = [tf.reduce_mean(analyzer_output, -1) for analyzer_output in analyzer_outputs]
		
		predict_onehot = tf.stack(analyzer_outputs, axis=-1, name="predict_onehot")
		predict_accent = tf.argmax(predict_onehot, axis=-1, output_type=tf.int32, name="predict_accent")
	
	with tf.name_scope("train"):
		correct_rate = tf.reduce_mean(tf.cast(tf.equal(accent, predict_accent), tf.float32), name="correct_rate")
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=accent_onehot, logits=predict_onehot, name="loss")
		loss_sum = tf.reduce_sum(loss, name="loss_sum")
		loss_mean = tf.reduce_mean(loss, name="loss_mean")
		
		training_step = tf.Variable(0, trainable=False, name="training_step")
		training_rate = tf.Variable(0.001, trainable=False, name="training_rate")
		training = tf.train.AdamOptimizer(training_rate).minimize(loss_sum, training_step, name="training")
	
	# init
	sess.run(tf.global_variables_initializer())


def save_graph(sess, graph_file):
	for dir_path in [conf.root_dir, conf.log_dir, conf.graph_dir]:
		os.makedirs(dir_path, exist_ok=True)
	sess = tf.get_default_session()
	tf.train.Saver().save(sess, graph_file)
	print("graph saved")


def load_graph(sess, meta_file, graph_dir):
	tf.train.import_meta_graph(meta_file).restore(sess, tf.train.latest_checkpoint(graph_dir))
	print("graph loaded")


def train_graph(sess, words_data, accents_data, batch_size=10, repeat_count=1000):
	summary_writer = tf.summary.FileWriter(conf.log_dir)
	summary_loss = tf.summary.scalar("summary_loss", sess.graph.get_tensor_by_name("train/loss_mean:0"))
	for i in range(repeat_count):
		batch_index = np.random.randint(len(words_data), size=[batch_size])
		val_word = words_data.take(batch_index, axis=0)
		val_accent = accents_data.take(batch_index, axis=0)
		val_training, val_loss, val_training_step, val_summary_loss = sess.run(
			["train/training", "train/loss:0", "train/training_step:0", summary_loss],
			{"input/word:0": val_word, "input/accent:0": val_accent, "train/training_rate:0": 0.002}
		)
		print("[", val_training_step, "] train: loss=", np.mean(val_loss))
		summary_writer.add_summary(val_summary_loss, val_training_step)
	print("trained", repeat_count, "times")


def test_graph(sess, words_data, accents_data, test_count=200):
	batch_index = np.random.randint(len(words_data), size=[test_count])
	val_word = words_data.take(batch_index, axis=0)
	val_accent = accents_data.take(batch_index, axis=0)
	val_correct_rate, val_training_step = sess.run(["train/correct_rate:0", "train/training_step:0"], {
		"input/word:0": val_word,
		"input/accent:0": val_accent,
	})
	print("[", val_training_step, "] test: correct_rate=", val_correct_rate * 100, "%")


def use_graph(sess, words_data):
	words_data = np.array(words_data, dtype=np.int32)
	return sess.run("analyzer/predict_accent:0", {
		"input/word:0": words_data,
	})
