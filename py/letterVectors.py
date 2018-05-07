import matplotlib.pylab as plt

from contexts.graphContext import *

load_graph()

letters = conf.letters
vectors = session().run("input/letter_vector_map:0")

xs = vectors[:, 2]
ys = vectors[:, 1]

for i in range(len(letters)):
	plt.annotate(letters[i], xy=(xs[i], ys[i]))
plt.show()
