import io

import numpy as np

import conf
import tools

# load data
data_file = io.open(conf.standardData_file, encoding="utf8")
lines = data_file.readlines()

words_data = []
accentsData = []
for line in lines:
	elements = line.split()
	word = str(elements[0])
	accent = int(elements[1])
	
	word_data = tools.parse_word(word)
	if word_data is None:
		continue
	
	words_data.append(word_data)
	accentsData.append(accent)

words_data = np.array(words_data)
accentsData = np.array(accentsData)
np.savez(conf.cookedData_file, words=words_data, accents=accentsData)
print("cookedData saved")
