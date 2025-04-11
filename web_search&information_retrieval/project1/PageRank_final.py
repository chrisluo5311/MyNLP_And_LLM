import sys
import numpy as np
import time

def read_input_file(file, d_weight):
    page_dict = {}
    s = set()
    with open(file, 'r') as f:
        for line in f:
            split_arr = line.split(":")
            source = int(split_arr[0])
            destination = split_arr[1].split(",")
            destination = [int(s.strip()) for s in destination]
            # print(source,":", destination)
            page_dict[source] = destination
            s.add(source)
            for des in destination:
                s.add(des)
    return rank_page(page_dict,d_weight, len(s))

def rank_page(page_dict, d, n, error = 0.0000000001):
    M = np.zeros((n,n))
    for index in range(n):
        if index in page_dict.keys():
            len_destination = len(page_dict[index])
            for des in page_dict[index]:
                M[des,index] = round(1/len_destination,10)
        else:
            for row in range(n):
                M[row, index] = 1/n

    # initialize v
    np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})
    v_old = np.full((n,1), 1/n)
    teleport = (1-d)*np.full((n,1), 1/n)

    # do a first round to get a v_new
    time_each_run = []
    start_time = time.time()
    v_new = d*(np.matmul(M,v_old)) + teleport
    time_each_run.append(time.time() - start_time)

    # keep calculating pagerank until convergence
    total_run = 1
    while np.linalg.norm(v_new - v_old, 2) > error:
        start_time = time.time()
        v_old = v_new
        v_new = d*(np.matmul(M,v_old)) + (1-d)*1/n
        time_each_run.append(time.time() - start_time)
        total_run += 1

    for i, vn in enumerate(v_new):
        print(f"p({i})={vn[0]:.10f}")
    return np.average(time_each_run)

if __name__ == '__main__':
    input_file_name = sys.argv[1]
    d_weight_1 = float(sys.argv[2])
    read_input_file(input_file_name, d_weight_1)


