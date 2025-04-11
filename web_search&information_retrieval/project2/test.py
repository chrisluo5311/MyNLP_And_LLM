import numpy as np
import os
import random
from collections import Counter
from typing import List

def delete(data):
    data2 = np.delete(data, 0, axis=0)
    for i in range(data2.shape[0]):
        print(i)

def sort_arr(arr):
    arr2 = arr[:3]
    arr2 = sorted(arr2, reverse=True)
    print(arr2)

def get_file_name(file):
    file_name = os.path.basename(file)
    print("5" in file_name)
    return 5

def test_file_size(file1, file2):
    file1_len = os.path.getsize(file1)
    file2_len = os.path.getsize(file2)
    if file1_len == file2_len:
        print(f"{file1_len} : {file2_len} - same size!")
    else:
        print(f"{file1_len} : {file2_len} - different size!")

def argsort_arr():
    arr = np.array([
        [3, -1, 4, -2],
        [-5, 8, 6, 0],
        [7, -3, -9, 2]
    ])

    sorted_indices = np.argsort(-np.abs(arr), axis=1)[:,:2]
    print(sorted_indices)

def check_file_format(file):
    print(f"Check file: {os.path.basename(file)}...")
    result = True
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                each_rate = line.split(" ")
                each_rate = [int(s.strip()) for s in each_rate]
                user = each_rate[0]
                movie = each_rate[1]
                ratings = each_rate[2]
                if ratings > 5 or ratings < 1:
                    result = False
                    print(f"Invalid format at user: {user}, movie: {movie}, rate: {ratings} ")
    if result:
        print("No ratings out of bound ✅")

def random_pick_from_arr(arr):
    # selected = random.sample(range(len(arr)), 3)
    selected = random.sample(arr, 3)
    print(selected)

def compute_avg_test(data):
    new_data = data.copy()
    avg_data = np.zeros((new_data.shape[0], 1))
    for i in range(new_data.shape[0]):
        mask = (new_data[i] == -1)
        new_data[i] = np.where(mask, 0, new_data[i])
        non_zero_values = new_data[i][new_data[i] > 0]
        avg = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0
        # avg = np.average(new_data[i])
        avg_data[i] = avg
    return avg_data

def compute_IUF_test(item_data):
    mask = (item_data == -1)
    new_item_data = np.where(mask, 0, item_data).copy()
    each_col_avg = np.mean(new_item_data, axis=0)
    each_col_users_rated = np.sum(new_item_data > 0, axis=0)
    division_result = np.divide(each_col_avg, each_col_users_rated,
                                out=np.zeros_like(each_col_avg,dtype=float), where=each_col_users_rated!=0)
    IUF = np.zeros_like(division_result, dtype=float)
    positive_mask = (division_result > 0)
    IUF[positive_mask] = np.log(division_result[positive_mask])
    return IUF

def compute_pearson_weight(u1, u1_avg, u2, u2_avg):
    # copy and change all the -1 to 0
    u1_copy = u1.copy()
    u2_copy = u2.copy()
    mask = (u1_copy == -1) | (u2_copy == -1)
    u1_copy = np.where(mask, 0, u1_copy)
    u2_copy = np.where(mask, 0, u2_copy)
    # subtract average and do dot product
    u1_copy = u1_copy - u1_avg
    u2_copy = u2_copy - u2_avg
    nominator = np.dot(u1_copy, u2_copy)
    # divided by their norm
    denominator = np.linalg.norm(u1_copy, 2) * np.linalg.norm(u2_copy,2)
    result = 0
    if denominator != 0:
        result = nominator/denominator
    return result

def library_pearson_weight(u1,u2):
    cor = np.corrcoef(u1, u2)[0, 1]
    # print("Pearson Correlation (NumPy):", cor)
    return cor


def majority_vote_matrix(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Computes the majority vote for each element position across multiple matrices.

    :param matrices: List of 2D numpy arrays (matrices) with the same shape.
    :return: A single 2D numpy array where each element is determined by majority vote.
    """
    if not matrices:
        raise ValueError("Input list of matrices is empty.")

    # Ensure all matrices have the same shape
    shape = matrices[0].shape
    if not all(mat.shape == shape for mat in matrices):
        raise ValueError("All matrices must have the same dimensions.")

    # Convert list of matrices into a 3D numpy array
    stack = np.array(matrices)  # Shape: (num_matrices, rows, cols)

    # Function to apply majority voting for each position
    def majority_vote(values):
        count = Counter(values)
        most_common, freq = count.most_common(1)[0]
        return most_common

    # Apply majority vote along the first axis (i.e., across matrices)
    majority_matrix = np.apply_along_axis(majority_vote, axis=0, arr=stack)

    return majority_matrix



if __name__ == '__main__':
    # x = np.array([[1,2,3],
    #             [4,5,6],
    #             [7,8,9]])
    # delete(x)
    # arr = [2,4,3,5,7,1,8]
    # sort_arr(arr)

    # get file name
    # test5_index = get_file_name("test5.txt")
    # print(test5_index)

    # x = np.nan
    # print(x)
    # print(np.isnan(x))
    # x = 10
    # x = np.nan_to_num(x, nan=0.0)
    # print(x)

    # test = np.array([[1,2,3],
    #                  [4,5,6],
    #                  [7,8,9]])
    # IUF = compute_IUF_test(test)
    # print(IUF)


    path = "./JIDUNG_result/Method8/new_10/"
    test_file_size("./example_result/example_result5.txt", path+"result_of_5_method_8_weighted_ratings.txt")
    test_file_size("./example_result/example_result10.txt", path+"result_of_10_method_8_weighted_ratings.txt")
    test_file_size("./example_result/example_result20.txt", path+"result_of_20_method_8_weighted_ratings.txt")
    #
    # # argsort_arr()
    # # print(abs(-1))
    check_file_format(path+"result_of_5_method_8_weighted_ratings.txt")
    check_file_format(path+"result_of_10_method_8_weighted_ratings.txt")
    check_file_format(path+"result_of_20_method_8_weighted_ratings.txt")

    # random_pick_from_arr([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    # ratings = np.array([
    #     [4, 5, 3, 4, 2],  # User a
    #     [5, 3, 4, 4, 1],  # User u
    #     [3, 4, 2, 5, 3]  # User v
    # ])
    #
    # corr_au = library_pearson_weight(ratings[0], ratings[1])
    # corr_av = library_pearson_weight(ratings[0], ratings[2])
    # print("Pearson Correlation between a and u (NumPy):", corr_au)
    # print("Pearson Correlation between a and v (NumPy):", corr_av)
    #
    # avg_data = compute_avg_test(ratings)
    # own_au = compute_pearson_weight(ratings[0], avg_data[0], ratings[1], avg_data[1])
    # own_av = compute_pearson_weight(ratings[0], avg_data[0], ratings[2], avg_data[2])
    # print("Pearson Correlation between a and u (Own):", own_au)
    # print("Pearson Correlation between a and v (Own):", own_av)
    #
    # if own_au == corr_au and own_av == corr_av:
    #     print("Same result✅")
    # else:
    #     print("Different result❗️")

    # matrix1 = np.array([[1, 4, 2], [3, 3, 1]])
    # matrix2 = np.array([[2, 2, 2], [3, 3, 3]])
    # matrix3 = np.array([[1, 4, 1], [3, 3, 1]])
    #
    # matrices = [matrix1, matrix2, matrix3]
    # result_matrix = majority_vote_matrix(matrices)
    #
    # print("Majority Vote Matrix:")
    # print(result_matrix)

    # init_weight = np.random.rand(6)
    # print(init_weight)