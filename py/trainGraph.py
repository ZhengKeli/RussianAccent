from contexts.trainContext import *

for g in range(20):
	for i in range(10):
		train_graph(500, 20)
		test_graph(500)
	save_graph()

# train_graph()
# test_graph()
# use_graph()
# save_graph()
