import os

import tensorflow as tf

import conf

# arguments
word_length = conf.word_length
letter_count = conf.letter_count
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
	
	predict_onehot = tf.stack(analyzer_outputs, -1, name="predict_onehot")
	predict_accent = tf.argmax(predict_onehot, -1, output_type=tf.int32, name="predict_accent")

with tf.name_scope("train"):
	correct_rate = tf.reduce_mean(tf.cast(tf.equal(accent, predict_accent), tf.float32), name="correct_rate")
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=accent_onehot, logits=predict_onehot, name="loss")
	loss_sum = tf.reduce_sum(loss, name="loss_sum")
	loss_mean = tf.reduce_mean(loss, name="loss_mean")
	
	training_step = tf.Variable(0, trainable=False, name="training_step")
	training_rate = tf.Variable(0.001, trainable=False, name="training_rate")
	training = tf.train.AdamOptimizer(training_rate).minimize(loss_sum, training_step, name="training")

# init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# summary and saving graph
for dir_path in [conf.root_dir, conf.log_dir, conf.graph_dir]:
	os.makedirs(dir_path, exist_ok=True)
tf.summary.FileWriter(conf.log_dir).add_graph(sess.graph)
tf.train.Saver().save(sess, conf.graph_file)
print("graph saved")
