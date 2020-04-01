"""
responsible for loading data in based on memory availability
"""
import numpy as np
import os
from io import StringIO

class DataLoader():

	def __init__(self):
		pass

	@staticmethod
	def load_file(data_dir, file_path, formats, format_type="csv"):
		if(format_type == "csv"):
			file_to_open = data_dir + "/" + file_path
			with open(file_to_open) as in_f:
				headings = in_f.readline()
			headings = headings.split(",")
			headings = map(lambda e: e.strip(), headings)

			headings = tuple(headings)
			print(headings)

			d = StringIO(u"M 21 72\nF 35 58")
			arr = np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
								                       'formats': ('S1', 'i4', 'f4')})
			data = np.loadtxt(file_to_open, delimiter=",",
							  comments="D", dtype={'names': headings,
												   'formats':formats,
												   })
		return data

