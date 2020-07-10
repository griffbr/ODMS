# Text logging functions.
def read_data_line(file_name, line_n):
	with open(file_name) as input_file:
		for i, line in enumerate(input_file):
			if i == line_n:
				data_list = line.split(" ")
			elif i > line_n:
				break
	return data_list
