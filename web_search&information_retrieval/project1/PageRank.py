import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

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
    # print("updated 𝑀:\n", M)
    plot_sparse_metrix(M)

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

    # for i, vn in enumerate(v_new):
    #     print(f"p({i})={vn[0]:.10f}")
    total_prob = check_prob_one(v_new)
    write_report(d, total_run, time_each_run, total_prob, v_new)
    return np.average(time_each_run)

def check_prob_one(v_new):
    total_prob = v_new.sum()
    print("✅total probability: ", total_prob)
    return total_prob

def write_report(d_factor, total_run, time_each_run, total_prob, v_new):
    with open("report.txt", "a") as f:
        f.write(f"1. d_weight: {d_factor}\n")
        f.write(f"2. Total run: {total_run}\n")
        f.write(f"3. Average time: {np.average(time_each_run)}\n")
        f.write(f"4. Total probability: {total_prob}\n")
        # put in dict and sort in des
        tmp_v_arr = {}
        for i, vn in enumerate(v_new):
            tmp_v_arr[i] = vn[0]
        sorted_vn = dict(sorted(tmp_v_arr.items(), key=lambda item: item[1], reverse=True))
        counter = 0
        f.write("====sorted====\n")
        for index, svn in sorted_vn.items():
            if 20 <= counter < 9980:
                counter += 1
                continue
            if counter == 0:
                f.write("====Top 20 sites====\n")
            elif counter == 9980:
                f.write("====Bottom 20 sites====\n")
            # print(f"p({index})={svn}")
            f.write(f"p({index})={svn}\n")
            counter += 1
        f.write("\n")

def plot_avg_time(avg_time_array):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    x = np.arange(0.75, 1, 0.05)
    print(x)
    plt.title('Average time per d')
    plt.xlabel("d_weight")
    plt.ylabel("Average time")
    plt.plot(x, avg_time_array, marker='o')
    plt.savefig("avg_per_d.jpg")

def plot_sparse_metrix(matrix):
    cal_sparse(matrix)
    plt.figure(figsize=(10, 10))
    # 計算分位數
    quantiles = np.quantile(matrix, np.linspace(0, 1, 10))  # 將數據分成 10 個分位數

    # 創建基於分位數的顏色映射
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(quantiles, cmap.N)
    plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')

    plt.colorbar(label='Value')
    plt.title('Sparsity of M')
    plt.xlabel('Column Pages')
    plt.ylabel('Row Pages')
    plt.savefig("matrix_M.jpg")
    plt.show()

def cal_sparse(matrix):
    print("matrix:\n",matrix)
    total_elements = matrix.size
    zero_values = np.count_nonzero(matrix == 0)

    # Calculate sparsity
    sparsity = zero_values / total_elements

    print(f"Total elements: {total_elements}")
    print(f"Number of zero values: {zero_values}")
    print(f"Sparsity: {sparsity:.2f}")

    if sparsity > 0.5:
        print("✅The dataset is sparse.")
    else:
        print("❌The dataset is not sparse.")

if __name__ == '__main__':
    test = "input2.txt"
    prod = "input.txt"
    # input_file_name = sys.argv[1]
    # d_weight_1 = float(sys.argv[2])
    # read_input_file(input_file_name, d_weight_1)

    # test
    open("report.txt", "w").close()
    # [0.75, 0.8, 0.85, 0.9, 0.95]
    d_weight = list(np.round(np.linspace(0.75, 0.95, num=5), decimals=2))
    avg_time_array = np.array([])
    for d in d_weight:
        avg_time_per_d = read_input_file(prod, d)
        avg_time_array = np.append(avg_time_array, avg_time_per_d)

    # plot avg time per d
    plot_avg_time(avg_time_array)