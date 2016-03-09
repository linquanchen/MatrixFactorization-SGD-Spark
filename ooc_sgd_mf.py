#!/usr/bin/env python3

# By Jinliang Wei (jinlianw@cs.cmu.edu)
# Copyright (c) 2016 Carnegie Mellon University
# For use in 15-719 only
# All other rights reserved.

# A simplistic sequential out-of-core program to perform matrix factorization using
# Stochastic Gradient Descent (SGD).
# The input data matrix is factorized into two lower rank matrices, W and H.
# Given that the data matrix and the two factor matrices could be too large to
# to fit in main memory, we process a piece, .i.e a block, at a time.
# The data and the intermediate states are stored as text files for easy human inspection
# and its effect on precision is negligible (compared to binary file).

# The program takes in a list a arguments:
# 1) the file containing the data matrix to be factorized, in CSV format
# 2) rank of the factor matrices
# 3) N - see explanation below
# 4) temporary scratch space

# The data matrix is partitioned into N*N pieces and the factor matrices W and H
# are each partitioned in to N pieces. The program holds 1 piece of the data
# matrix and 1 piece of each factor matrices during SGD computation.

# example command to run:
# $ ./ooc_sgd_mf.py ratings.csv 16 2 tmp

import sys
import numpy as np
import os
import random
import time

data_block_file = "block.dat"
W_filename = "W.csv"
H_filename = "H.csv"
split_sign = ","
# blockify the data matrix
def blockify_data(csv_file, tmp_dir, N):
    max_x_id = 0
    max_y_id = 0
    with open(csv_file) as fobj:
        for line in fobj:
            tokens = line.split(split_sign)
            max_x_id = max(max_x_id, int(tokens[0]))
            max_y_id = max(max_y_id, int(tokens[1]))
    print(str(max_x_id) + " " + str(max_y_id))
    
    # assume the id starts from 0
    x_block_dim = int((max_x_id + N) / N)
    y_block_dim = int((max_y_id + N) / N)
    
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # create temporary data files
    tmp_files = []
    for i in range(0, N):
        files = []
        for j in range(0, N):
            fobj = open(tmp_dir + "/data-" + str(i) + "-" + str(j) + ".csv", 'w+')
            files.append(fobj)
        tmp_files.append(files)
    
    with open(csv_file) as fobj:
        for line in fobj:
            tokens = line.split(split_sign)
            x_id = int(tokens[0])
            y_id = int(tokens[1])
            x_block_id = int(x_id / x_block_dim)
            y_block_id = int(y_id / y_block_dim)
            #print (x_block_id, y_block_id, x_id, y_id)
            tmp_files[x_block_id][y_block_id].write(line)

    for i in range(0, N):
        for j in range(0, N):
             tmp_files[i][j].close()
    
    block_fobj = open(data_block_file, 'w+')
    for i in range(0, N):
        for j in range(0, N):
            out_line = ""
            with open(tmp_dir + "/data-" + str(i) + "-" + str(j) + ".csv", 'r') as fobj:
                for line in fobj:
                    tokens = line.split(split_sign)
                    out_line += tokens[0] + "," + tokens[1] + "," + str(float(tokens[2])) + ";"
            block_fobj.write(out_line + "\n")
    block_fobj.close()

    return x_block_dim, y_block_dim

# randomly initialize the factor matrices
def initialize_factor_matrices(N, K, x_block_dim, y_block_dim):
    random.seed(1) # always use the same seed to get deterministic results
    fobj = open(W_filename, 'w+')
    for i in range(0, N):
        line = ""
        for j in range(0, x_block_dim):
            row = ""
            for k in range(0, K):
                row += str(random.random()) + ","
            line += row +";"
        fobj.write(line + "\n")
    fobj.close()

    fobj = open(H_filename, 'w+')
    for i in range(0, N):
        line = ""
        for j in range(0, y_block_dim):
            row = ""
            for k in range(0, K):
                row += str(random.random()) + ","
            line += row +";"
        fobj.write(line + "\n")
    fobj.close()

# one line is a partition of the factor matrix
def parse_factor_matrix_line(line):
    tokens = line.split(";")
    rows = []
    for token in tokens:
        row = []
        token = token.strip()
        if token == "":
            continue
        row_entries = token.split(",")
        for row_entry in row_entries:
            if row_entry == "":
                continue
            row.append(float(row_entry))
        rows.append(np.array(row))
    return rows

def sgd_on_one_block(data_line, W_rows, W_rows_offset,
                     H_rows, H_rows_offset, step_size):
    data_line = data_line.strip()
    data_samples = data_line.split(";")
    num_data_samples = 0
    
    for data_sample in data_samples:
        if data_sample == "":
            continue
        tokens = data_sample.split(",")
        x_id = int(tokens[0])
        y_id = int(tokens[1])
        rating = float(tokens[2])
        
        diff = rating - np.dot(W_rows[x_id - W_rows_offset], H_rows[y_id - H_rows_offset])
        W_gradient = -2 * diff * H_rows[y_id - H_rows_offset]
        W_rows[x_id - W_rows_offset] -= step_size * W_gradient
        
        H_gradient = -2 * diff * W_rows[x_id - W_rows_offset]
        H_rows[y_id - H_rows_offset] -= step_size * H_gradient
        num_data_samples += 1
        return num_data_samples

def factor_matrix_rows_to_string(rows):
    line = ""
    for row in rows:
        for num in np.nditer(row):
            line += str(num) + ","
        line += ";"
    return line

# perform the SGD algorithm one block at a time
def sgd_block_by_block(N, step_size, x_block_dim, y_block_dim):
    block_fobj = open(data_block_file, 'r')
    W_fobj = open(W_filename, 'r')
    W_new_fobj = open(W_filename + ".new", 'w+')
    
    for i in range(0, N):
        W_line = W_fobj.readline()
        W_rows = parse_factor_matrix_line(W_line)
        W_rows_offset = i * x_block_dim
        H_fobj = open(H_filename, 'r')
        H_new_fobj = open(H_filename + ".new", 'w+')
        
        for j in range(0, N):
            data_line = block_fobj.readline()
            H_line = H_fobj.readline()
            H_rows = parse_factor_matrix_line(H_line)
            H_rows_offset = j * y_block_dim
            
            sgd_on_one_block(data_line, W_rows, W_rows_offset,
                             H_rows, H_rows_offset, step_size)
                
            line = factor_matrix_rows_to_string(H_rows)
            H_new_fobj.write(line + '\n')
        
        H_new_fobj.close()
        H_fobj.close()
        os.remove(H_filename)
        os.rename(H_filename + ".new", H_filename)
        line = factor_matrix_rows_to_string(W_rows)
        W_new_fobj.write(line + '\n')
    W_fobj.close()
    W_new_fobj.close()
    block_fobj.close()
    os.remove(W_filename)
    os.rename(W_filename + ".new", W_filename)

# evaluate the current model on one block of the data
def evaluate_on_one_block(data_line, W_rows, W_rows_offset,
                          H_rows, H_rows_offset):
    data_line = data_line.strip()
    data_samples = data_line.split(";")
    error = .0
    n = 0
    
    for data_sample in data_samples:
        if data_sample == "":
            continue
        tokens = data_sample.split(",")
        x_id = int(tokens[0])
        y_id = int(tokens[1])
        rating = float(tokens[2])
        
        diff = rating - np.dot(W_rows[x_id - W_rows_offset], H_rows[y_id - H_rows_offset])
        error += diff ** 2
        n += 1
    
    return error, n

# perform evaluation of the model one block at a time
def evaluate_block_by_block(N, x_block_dim, y_block_dim):
    block_fobj = open(data_block_file, 'r')
    W_fobj = open(W_filename, 'r')
    error_total = .0
    n_total = 0
    
    # iterate over rows
    for i in range(0, N):
        W_line = W_fobj.readline()
        W_rows = parse_factor_matrix_line(W_line)
        W_rows_offset = i * x_block_dim
        H_fobj = open(H_filename, 'r')
        
        # iterate over blocks on the same row
        for j in range(0, N):
            data_line = block_fobj.readline()
            H_line = H_fobj.readline()
            H_rows = parse_factor_matrix_line(H_line)
            H_rows_offset = j * y_block_dim
            
            err, n = evaluate_on_one_block(data_line, W_rows, W_rows_offset,
                                           H_rows, H_rows_offset)
            error_total += err
            n_total += n
    return error_total, n_total

if __name__ == "__main__":
    print ("Out-of-Core SGD Matrix Factorization begins...")
    csv_file = sys.argv[1]
    K = int(sys.argv[2]) #rank
    N = int(sys.argv[3])
    tmp_dir = sys.argv[4]
    num_iterations = 10
    eta = 0.001
    eta_decay = 0.99
    
    x_block_dim, y_block_dim = blockify_data(csv_file, tmp_dir, N)
    print("Done partitioning data matrix...")
    
    initialize_factor_matrices(N, K, x_block_dim, y_block_dim)
    print("Done initializing factor matrices...")
    
    print("Start Stochastic Gradient Descent...")
    print("iteration", " seconds", " squared_error", " RMSE")
    t1 = time.clock()
    for i in range(0, num_iterations):
        sgd_block_by_block(N, eta, x_block_dim, y_block_dim)
        eta *= eta_decay
        
        error, n = evaluate_block_by_block(N, x_block_dim,
                                           y_block_dim)
        t2 = time.clock()
        print (i, int(t2 - t1), error, np.sqrt(error / n))
    
    print("Stochastic Gradient Descent Done, computation time =", int(t2 - t1), "seconds, exit now")
