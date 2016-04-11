import sys
import numpy as np

def load_factor_matrix(filename):
	return np.genfromtxt(filename,delimiter=',')

def load_data_matrix(filename):
	data_mat = list()
	with open(filename, "r") as data_in:
		for line in data_in:
			tokens = line.split(',')
			data_mat.append((int(tokens[0]), int(tokens[1]), float(tokens[2])))
	return data_mat

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print "python %s <w.csv> <h.csv> <d.csv>" % sys.argv[0]
		exit(1)

	w_mat = load_factor_matrix(sys.argv[1])
	h_mat = load_factor_matrix(sys.argv[2])
	d_mat = load_data_matrix(sys.argv[3])

	# calculate RMSE
	error, n = 0.0, 0
	for entry in d_mat:
		row, col, score = entry
		pred = np.dot(w_mat[row], h_mat[col])
		error += (score - pred) ** 2
		n += 1
	rmse = np.sqrt(error/n)
	print "Calculated_RMSE=%f" % rmse
