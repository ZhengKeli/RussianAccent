import matplotlib.pylab as plt

from contexts.graphContext import *

letters = conf.letters
vectors = sess.run("input/letter_vector_map:0")

xs = vectors[:, 2]
ys = vectors[:, 1]

for i in range(len(letters)):
	plt.annotate(letters[i], xy=(xs[i], ys[i]))
plt.show()
