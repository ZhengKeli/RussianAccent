import numpy as np
import tensorflow as tf

import conf
import tools

# load graph
sess = tf.Session()
tf.train.import_meta_graph(conf.graph_meta_file) \
	.restore(sess, tf.train.latest_checkpoint(conf.graph_dir))
print("graph loaded")

while True:
	# input words
	words = input("Type the word(s):\n")
	if words == "exit":
		break
	words = words.split()
	
	# cook data
	words_data = []
	for word in words:
		word_data = tools.parse_word(word)
		if word_data is not None:
			words_data.append(word_data)
	words_data = np.array(words_data, dtype=np.int32)
	
	# run graph
	val_predict_accent = sess.run("analyzer/predict_accent:0", {
		"input/word:0": words_data,
	})
	
	# print
	print("Prediction:")
	for i in range(len(words)):
		print_string = words[i]
		insert_index = val_predict_accent[i] + 1
		print_string = print_string[0:insert_index] + "'" + print_string[insert_index:len(print_string)]
		print(print_string)
	print()

print("bye!")
