import matplotlib.pylab as plt
import tensorflow as tf

import conf

# load graph
sess = tf.Session()
tf.train.import_meta_graph(conf.graph_meta_file) \
	.restore(sess, tf.train.latest_checkpoint(conf.graph_dir))
print("graph loaded")

# read map
letters = conf.letters
vectors = sess.run("input/letter_vector_map:0")

xs = vectors[:, 2]
ys = vectors[:, 1]

for i in range(len(letters)):
	plt.annotate(letters[i], xy=(xs[i], ys[i]))
plt.show()
