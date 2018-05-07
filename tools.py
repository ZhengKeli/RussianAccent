import numpy as np

import conf


def parse_word(word):
	if len(word) > conf.word_length:
		return None
	
	word_data = np.zeros([conf.word_length], np.int32)
	for i in range(conf.word_length):
		if i >= len(word):
			break
		
		letter_data = conf.letters.find(word[i])
		if letter_data == -1:
			return None
		
		word_data[i] = letter_data
	return word_data
