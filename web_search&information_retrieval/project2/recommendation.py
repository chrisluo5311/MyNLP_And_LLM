import os
from collections import Counter
import numpy as np
import random

# Method name for predicting train data => not using anymore ❌
COSINE_SIMILAR = "cosine-similarity"
PEARSON_CORRELATION = "pearson-correlation"

# File path for storing results
RESULT_FILE_PATH = "./JIDUNG_result/"

# Pearson correlation amplification
AMPLIFICATION = 2.5

# Namings for splitting train data usage
train_split_1 = "train_split_1.txt"
train_split_ground_truth = "train_split_ground_truth.txt"
train_split_to_test = "train_split_to_test.txt"
train_split_to_test_mix = "train_split_to_test_mix.txt"
train_split_to_test_3 = "train_split_to_test_3.txt"

def cos_similarity(u1, u2):
    new_u1 = u1.copy()
    new_u2 = u2.copy()
    mask = (new_u1 == -1) | (new_u2 == -1)
    new_u1 = np.where(mask, 0, new_u1)
    new_u2 = np.where(mask, 0, new_u2)
    nominator = np.dot(new_u1, new_u2)
    denominator = np.linalg.norm(new_u1, 2) * np.linalg.norm(new_u2, 2)
    result = 0
    if denominator != 0:
        result = nominator/denominator
    return result

# log of total # of users over total # of users rated movie j
# In item_data, -1 : unobserved, 1-5 : observed
def compute_IUF(item_data):
    mask = (item_data == -1)
    new_item_data = np.where(mask, 0, item_data).copy()
    total_users = new_item_data.shape[0]
    each_col_users_rated = np.sum(new_item_data > 0, axis=0)
    division_result = np.divide(total_users, each_col_users_rated,
                                out=np.zeros_like(each_col_users_rated,dtype=float), where=each_col_users_rated!=0)
    IUF = np.zeros_like(division_result, dtype=float)
    positive_mask = (division_result > 0)
    IUF[positive_mask] = np.log(division_result[positive_mask])
    return IUF

def compute_pearson_weight(u1, u1_avg, u2, u2_avg):
    mask = (u1 > 0) & (u2 > 0)
    u1_centered = np.zeros_like(u1, dtype=float)
    u2_centered = np.zeros_like(u2, dtype=float)
    u1_centered[mask] = u1[mask] - u1_avg
    u2_centered[mask] = u2[mask] - u2_avg
    nominator = np.dot(u1_centered, u2_centered)
    # divided by their norm
    denominator = np.linalg.norm(u1_centered, 2) * np.linalg.norm(u2_centered,2)
    result = 0
    if denominator != 0:
        result = nominator/denominator
    return result

def compute_pearson_weight_IUF(u1, u1_avg, u2, u2_avg, IUF):
    mask = (u1 > 0) & (u2 > -1)
    u1_centered = np.zeros_like(u1, dtype=float)
    u2_centered = np.zeros_like(u2, dtype=float)
    u1_centered[mask] = u1[mask] - u1_avg
    u2_centered[mask] = u2[mask] - u2_avg
    nominator = np.dot(IUF*u1_centered, u2_centered)
    # divided by their norm
    norm_u1 = np.sqrt(np.sum((IUF ** 2) * (u1_centered ** 2)))
    norm_u2 = np.sqrt(np.sum((IUF ** 2) * (u2_centered ** 2)))
    denominator = norm_u1 + norm_u2
    result = 0
    if denominator != 0:
        result = nominator/denominator
    return result

def compute_item_matrix(train_data, train_avg):
    mask = (train_data == -1)
    item_data = np.where(mask, 0, train_data).copy()
    item_data_T = item_data.T.copy()
    # subtract the user avg
    item_data = item_data - train_avg
    item_data_T = item_data_T - train_avg.T
    # norm
    item_data_norm = np.linalg.norm(item_data, ord=2, axis=0) # (1,1000)
    item_data_norm = item_data_norm.reshape(1, item_data.shape[1])
    item_data_T_norm = np.linalg.norm(item_data_T, ord=2, axis=1) # (1000,1)
    item_data_T_norm = item_data_T_norm.reshape(item_data_T.shape[0], 1)
    # print(f"item data norm: {item_data_norm.shape}")
    # print(f"item data_T norm: {item_data_T_norm.shape}")

    # dot product
    numerator = np.dot(item_data_T,item_data) # (1000,1000)
    denominator = np.dot(item_data_T_norm, item_data_norm) # (1000,1000)

    # element wise division => item metric
    result = numerator / denominator
    np.fill_diagonal(result, 0.0)
    # print(f"item metric: {result.shape}")
    return result

def top_k_similarity(data, target, k):
    cos_arr = []
    for i in range(data.shape[0]):
        # [{}, {} , ...]
        cos_arr.append({i: cos_similarity(target, data[i])})

    # sort the cos_arr based on the value(cos similarity)
    sorted_cos = sorted(cos_arr, key=lambda x:list(x.values())[0], reverse=True)
    return sorted_cos[:k]

# use all available ratings to compute average
def compute_avg(data):
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

# The result is actually not good => stop using
# def predict_train_data(train_data, top_k, method_name):
#     if method_name == COSINE_SIMILAR:
#         sim_pair = {}
#         for i in range(train_data.shape[0]):
#             each_train = train_data[i]
#             similar_arr = top_k_similarity(train_data, each_train, top_k)
#             print(f"User {i + 1}'s top {top_k} similar users:\n{similar_arr}")
#             # filter out self
#             filtered_sim_arr = [s for s in similar_arr if i not in s]
#             print(f"After filtering self: {filtered_sim_arr}")
#             sim_pair[i] = filtered_sim_arr
#
#         count_not_enough_user = 0
#         for u_index, pair_data in sim_pair.items():
#             each_train_u = train_data[u_index]
#             unobserved_indices = np.where(each_train_u == -1)[0]
#             print(f"Before: User: {u_index+1} There are {len(unobserved_indices)} unobserved movies")
#             for j in unobserved_indices:
#                 nominator = 0
#                 denominator = 0
#                 for each_u in pair_data:
#                     # {46: 0.234...}
#                     u_id = list(each_u.keys())[0]
#                     train_user_data = train_data[u_id]
#                     weight = list(each_u.values())[0]
#                     if train_user_data[j] == -1:
#                         denominator += weight
#                         continue
#                     else:
#                         ratings = train_user_data[j]
#                     nominator += weight * ratings
#                     denominator += weight
#                 prediction = math.ceil(nominator / denominator)
#                 each_train_u[j] = prediction
#
#             unobserved_indices = np.where(each_train_u == -1)[0]
#             print(f"After: User: {u_index+1} There are {len(unobserved_indices)} unobserved movies")
#             if np.any(each_train_u == -1) or np.any(each_train_u == 0):
#                 count_not_enough_user += 1
#                 print(f"top {top_k} users is not enough to predict user {u_index+1}\n")
#         print(f"❗️️️❗️Total {count_not_enough_user} of users not fully predicted❗️❗️")
#     elif method_name == PEARSON_CORRELATION:
#         train_avg = compute_avg(train_data)
#         weight_arr = np.zeros((train_data.shape[0], train_data.shape[0]))  # 200 x 200
#         # for i in range(train_data.shape[0]):
#         #     each_train = train_data[i]
#         #     for j in range(train_data.shape[0]):
#         #         # if i != j:
#
#
#     return train_data


def method1_cosine(test_data, test_index, test_avg, train_data, top_k):
    sim_pair = {}
    item_avg = compute_item_avg(train_data)
    for i in range(test_data.shape[0]):
        # get top k similar users from train based on each test data
        each_test_data = test_data[i]
        similar_arr = top_k_similarity(train_data, each_test_data, top_k)
        # test_index: 0, 200, 300, 400
        print(f"User {test_index+i+1}'s top {top_k} similar users:\n{similar_arr}")
        sim_pair[i] = similar_arr
    # predict the 0 in test data with the paired users' data in train
    # sim_pair: { 0 : [{46: 0.234...}, {172, 0.432...}, ...], 1 : ... }
    count_not_enough_user = 0
    print(f"======= Result of choosing top {top_k} neighbors =======")
    test_data = test_data.astype(float) # todo: comment this out for testing individually
    for test_user_index, pair_data in sim_pair.items():
        each_test_user_data = test_data[test_user_index]
        unobserved_indices = np.where(each_test_user_data == 0)[0]
        print(f"Before: User: {test_user_index+1+test_index} There are {len(unobserved_indices)} unobserved ratings")
        # search the train data user and supplement the unobserved data if exist
        for j in unobserved_indices:
            nominator = 0
            denominator = 0
            for each_train_user in pair_data:
                # top k users' similarity {userid: cos...}
                train_user_id = list(each_train_user.keys())[0]
                train_user_data = train_data[train_user_id]
                weight = list(each_train_user.values())[0]
                if train_user_data[j] == -1:
                    # unobserved data: ignore
                    continue
                ratings = train_user_data[j]
                nominator += weight*ratings
                denominator += weight

            prediction = 0
            if denominator != 0:
                # prediction = round(nominator / denominator) #todo: open this up for testing individually
                prediction = nominator / denominator # todo: open this up for testing method 8
            if prediction == 0:
                prediction = test_avg[test_user_index][0]
                # prediction = item_avg[j]
            prediction = max(1, min(5, prediction))
            each_test_user_data[j] = prediction

        unobserved_indices = np.where(each_test_user_data == 0)[0]
        print(f"After: User: {test_user_index+1+test_index} There are {len(unobserved_indices)} unobserved ratings")
        if np.any(each_test_user_data == 0):
            count_not_enough_user += 1
            print(f"top {top_k} users is not enough to predict user {test_user_index+1+test_index}\n")
    print(f"❗️️️❗️Total {count_not_enough_user} of users not fully predicted❗️❗️")
    return test_data

def method2_pearson_correlation(test_data, test_index, train_data, top_k):
    # Compute avg for test and train
    train_avg = compute_avg(train_data) # (200,)
    test_avg = compute_avg(test_data) # (100,)
    item_avg = compute_item_avg(train_data)
    # compute weight
    weight_arr = np.zeros((test_data.shape[0], train_data.shape[0])) # 100 x 200
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        each_test_avg = test_avg[i]
        for j in range(train_data.shape[0]):
            each_train = train_data[j]
            each_train_avg = train_avg[j]
            # weight range: -1 0 1
            weight_arr[i][j] = compute_pearson_weight(each_test, each_test_avg, each_train, each_train_avg)
    # sort the weight and take top k users
    sorted_train_users_indices = np.argsort(-np.abs(weight_arr), axis=1)[:,:top_k]

    # predict p for each test users
    count_not_enough_user = 0
    test_data = test_data.astype(float) # todo: comment this out for testing individually
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        unobserved_indices = np.where(each_test == 0)[0]
        print(f"Before: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings")
        for j in unobserved_indices:
            nominator = 0
            denominator = 0
            for train_user_index in sorted_train_users_indices[i]:
                w = weight_arr[i][train_user_index]
                if train_data[train_user_index][j] == -1:
                    # unobserved data: skip
                    continue
                r = train_data[train_user_index][j]
                r_avg_train = train_avg[train_user_index][0]
                nominator += (r - r_avg_train)*w
                denominator += abs(w)
            r_avg_test = test_avg[i][0]
            # apply pearson correlation formula
            user_diff = 0
            if denominator != 0:
                user_diff = nominator / denominator
            print(f"User {i + 1 + test_index} Movie {j+1} r_avg_test: {r_avg_test}, user_diff: {user_diff}")
            # prediction = round(r_avg_test + user_diff) #todo: open this up for testing individually
            prediction = r_avg_test + user_diff # todo: open this up for testing method 8
            print(f"Prediction: {prediction}")
            if prediction > 5 or prediction < 1:
                print(f"️️️❗️❗️️️❗️❗️️️❗️Prediction out of bound: {prediction}❗❗️❗️️️❗️❗️️️❗️")
                prediction = test_avg[i][0]
                # prediction = item_avg[j]
            prediction = max(1, min(5, prediction))
            print(f"After bound: {prediction}")
            each_test[j] = prediction

        unobserved_indices = np.where(each_test == 0)[0]
        print(f"After: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings\n")
        if np.any(each_test == 0):
            count_not_enough_user += 1
            print(f"top {top_k} users is not enough to predict user {i+1+test_index}\n")
    print(f"❗️️️❗️Total {count_not_enough_user} of users not fully predicted❗️❗️")
    return test_data

def method3_pearson_correlation_with_IUF(test_data, test_index, train_data, top_k, IUF):
    # Compute avg for test and train
    train_avg = compute_avg(train_data)  # (200,)
    test_avg = compute_avg(test_data)  # (100,)
    item_avg = compute_item_avg(train_data)
    # compute weight
    weight_arr = np.zeros((test_data.shape[0], train_data.shape[0]))  # 100 x 200
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        each_test_avg = test_avg[i]
        for j in range(train_data.shape[0]):
            each_train = train_data[j]
            each_train_avg = train_avg[j]
            # weight range: -1 0 1
            weight_arr[i][j] = compute_pearson_weight_IUF(each_test, each_test_avg, each_train, each_train_avg, IUF)
    # sort the weight and take top k users
    sorted_train_users_indices = np.argsort(-np.abs(weight_arr), axis=1)[:, :top_k]

    # predict p for each test users
    count_not_enough_user = 0
    test_data = test_data.astype(float) # todo: comment this out for testing individually
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        unobserved_indices = np.where(each_test == 0)[0]
        print(f"Before: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings")
        for j in unobserved_indices:
            nominator = 0
            denominator = 0
            for train_user_index in sorted_train_users_indices[i]:
                w = weight_arr[i][train_user_index]
                if train_data[train_user_index][j] == -1:
                    # unobserved data: skip
                    continue
                r = train_data[train_user_index][j]
                r_avg_train = train_avg[train_user_index][0]
                nominator += (r - r_avg_train) * w
                denominator += abs(w)
            r_avg_test = test_avg[i][0]
            # apply pearson correlation formula
            user_diff = 0
            if denominator != 0:
                user_diff = nominator / denominator
            print(f"User {i + 1 + test_index} Movie {j + 1} r_avg_test: {r_avg_test}, user_diff: {user_diff}")
            # prediction = round(r_avg_test + user_diff) #todo: open this up for testing individually
            prediction = r_avg_test + user_diff # todo: open this up for testing method 8
            print(f"Prediction: {prediction}")
            if prediction > 5 or prediction < 1:
                print(f"️️️❗️❗️️️❗️❗️️️❗️Prediction out of bound: {prediction}❗❗️❗️️️❗️❗️️️❗️")
                prediction = test_avg[i][0]
                # prediction = item_avg[j]
            prediction = max(1, min(5, prediction))
            print(f"After bound: {prediction}")
            each_test[j] = prediction

        unobserved_indices = np.where(each_test == 0)[0]
        print(f"After: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings\n")
        if np.any(each_test == 0):
            count_not_enough_user += 1
            print(f"top {top_k} users is not enough to predict user {i + 1 + test_index}\n")
    print(f"❗️️️❗️Total {count_not_enough_user} of users not fully predicted❗️❗️")
    return test_data

def method4_pearson_correlation_amplification(test_data, test_index, train_data, top_k):
    # Compute avg for test and train
    train_avg = compute_avg(train_data) # (200,)
    test_avg = compute_avg(test_data) # (100,)
    item_avg = compute_item_avg(train_data)
    # compute weight
    weight_arr = np.zeros((test_data.shape[0], train_data.shape[0])) # 100 x 200
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        each_test_avg = test_avg[i]
        for j in range(train_data.shape[0]):
            each_train = train_data[j]
            each_train_avg = train_avg[j]
            # weight range: -1 0 1
            weight_arr[i][j] = compute_pearson_weight(each_test, each_test_avg, each_train, each_train_avg)
    # amplification => to the power of 2.5 (typical choice)
    weight_arr = np.sign(weight_arr) * np.power(np.abs(weight_arr), AMPLIFICATION)

    # sort the weight and take top k users
    sorted_train_users_indices = np.argsort(-np.abs(weight_arr), axis=1)[:,:top_k]

    # predict p for each test users
    count_not_enough_user = 0
    test_data = test_data.astype(float) # todo: comment this out for testing individually
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        unobserved_indices = np.where(each_test == 0)[0]
        print(f"Before: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings")
        for j in unobserved_indices:
            nominator = 0
            denominator = 0
            for train_user_index in sorted_train_users_indices[i]:
                w = weight_arr[i][train_user_index]
                if train_data[train_user_index][j] == -1:
                    # unobserved data: skip
                    continue
                r = train_data[train_user_index][j]
                r_avg_train = train_avg[train_user_index][0]
                nominator += (r - r_avg_train)*w
                denominator += abs(w)
            r_avg_test = test_avg[i][0]
            # apply pearson correlation formula
            user_diff = 0
            if denominator != 0:
                user_diff = nominator / denominator
            print(f"User {i + 1 + test_index} Movie {j+1} r_avg_test: {r_avg_test}, user_diff: {user_diff}")
            # prediction = round(r_avg_test + user_diff) #todo: open this up for testing individually
            prediction = r_avg_test + user_diff # todo: open this up for testing method 8
            print(f"Prediction: {prediction}")
            if prediction > 5 or prediction < 1:
                print(f"️️️❗️❗️️️❗️❗️️️❗️Prediction out of bound: {prediction}❗❗️❗️️️❗️❗️️️❗️")
                prediction = test_avg[i][0]
                # prediction = item_avg[j]
            prediction = max(1, min(5, prediction))
            print(f"After bound: {prediction}")
            each_test[j] = prediction

        unobserved_indices = np.where(each_test == 0)[0]
        print(f"After: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings\n")
        if np.any(each_test == 0):
            count_not_enough_user += 1
            print(f"top {top_k} users is not enough to predict user {i+1+test_index}\n")
    print(f"❗️️️❗️Total {count_not_enough_user} of users not fully predicted❗️❗️")
    return test_data

def method5_pearson_correlation_with_IUF_AMP(test_data, test_index, train_data, top_k, IUF):
    # Compute avg for test and train
    train_avg = compute_avg(train_data)  # (200,)
    test_avg = compute_avg(test_data)  # (100,)
    item_avg = compute_item_avg(train_data)
    # compute weight
    weight_arr = np.zeros((test_data.shape[0], train_data.shape[0]))  # 100 x 200
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        each_test_avg = test_avg[i]
        for j in range(train_data.shape[0]):
            each_train = train_data[j]
            each_train_avg = train_avg[j]
            # weight range: -1 0 1
            weight_arr[i][j] = compute_pearson_weight_IUF(each_test, each_test_avg, each_train, each_train_avg, IUF)

    # amplification => to the power of 2.5 (typical choice)
    weight_arr = np.sign(weight_arr) * np.power(np.abs(weight_arr), AMPLIFICATION)

    # sort the weight and take top k users
    sorted_train_users_indices = np.argsort(-np.abs(weight_arr), axis=1)[:, :top_k]

    # predict p for each test users
    count_not_enough_user = 0
    test_data = test_data.astype(float) # todo: comment this out for testing individually
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        unobserved_indices = np.where(each_test == 0)[0]
        print(f"Before: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings")
        for j in unobserved_indices:
            nominator = 0
            denominator = 0
            for train_user_index in sorted_train_users_indices[i]:
                w = weight_arr[i][train_user_index]
                if train_data[train_user_index][j] == -1:
                    # unobserved data: skip
                    continue
                r = train_data[train_user_index][j]
                r_avg_train = train_avg[train_user_index][0]
                nominator += (r - r_avg_train) * w
                denominator += abs(w)
            r_avg_test = test_avg[i][0]
            # apply pearson correlation formula
            user_diff = 0
            if denominator != 0:
                user_diff = nominator / denominator
            print(f"User {i + 1 + test_index} Movie {j + 1} r_avg_test: {r_avg_test}, user_diff: {user_diff}")
            # prediction = round(r_avg_test + user_diff) #todo: open this up for testing individually
            prediction = r_avg_test + user_diff # todo: open this up for testing method 8
            print(f"Prediction: {prediction}")
            if prediction > 5 or prediction < 1:
                print(f"️️️❗️❗️️️❗️❗️️️❗️Prediction out of bound: {prediction}❗❗️❗️️️❗️❗️️️❗️")
                prediction = test_avg[i][0]
                # prediction = item_avg[j]
            prediction = max(1, min(5, prediction))
            print(f"After bound: {prediction}")
            each_test[j] = prediction

        unobserved_indices = np.where(each_test == 0)[0]
        print(f"After: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings\n")
        if np.any(each_test == 0):
            count_not_enough_user += 1
            print(f"top {top_k} users is not enough to predict user {i + 1 + test_index}\n")
    print(f"❗️️️❗️Total {count_not_enough_user} of users not fully predicted❗️❗️")
    return test_data

def compute_item_avg(train_data):
    new_data = train_data.copy()
    mask_matrix = np.ma.masked_where(new_data == -1, new_data)
    avg_data = np.mean(mask_matrix, axis=0)
    # avg_data = avg_data.reshape(1, -1)
    return avg_data

def method6_item_based_collaborative_filtering(test_data, test_index, train_data):
    # Compute avg for train
    train_avg = compute_avg(train_data)  # (200,)
    test_avg = compute_avg(test_data)  # (100,)
    item_avg = compute_item_avg(train_data)
    # Compute item-based weight (1000,1000) with adjusted cosine similarity
    item_weight_metric = compute_item_matrix(train_data, train_avg)
    # for every test user, predict unobserved ratings => 0 in test_data
    test_data = test_data.astype(float) # todo: comment this out for testing individually
    for i in range(test_data.shape[0]):
        each_test = test_data[i]
        each_test_avg = test_avg[i][0]
        unobserved_indices = np.where(each_test == 0)[0]
        observed_indices = np.where(each_test > 0)[0] # only five ratings
        print(f"Before: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings")
        # predict with item-based cf (adjusted cosine similarity)
        for j in unobserved_indices:
            numerator = 0
            denominator = 0
            for k in observed_indices:
                item_w = item_weight_metric[j][k]
                rate = each_test[k]
                numerator += item_w * (rate - each_test_avg)
                denominator += abs(item_w)

            prediction = 0
            print(item_avg[j])
            if denominator != 0:
                diff = numerator / denominator
                # prediction = round(each_test_avg + diff) #todo: open this up for testing individually
                prediction = each_test_avg + diff # todo: open this up for testing method 8

            if prediction > 5 or prediction < 1:
                print(f"️️️❗️❗️️️❗️❗️️️❗️Prediction out of bound: {prediction}❗❗️❗️️️❗️❗️️️❗️")
                prediction = item_avg[j]
            prediction = max(1, min(5, prediction))
            each_test[j] = prediction

        unobserved_indices = np.where(each_test == 0)[0]
        print(f"After: User: {i + 1 + test_index} There are {len(unobserved_indices)} unobserved ratings\n")

    return test_data

def method7_majority_voting(test_data, test_index, test_avg, train_data, dynamic_top_k, IUF):
    method1_result = method1_cosine(test_data, test_index, test_avg, train_data, dynamic_top_k[0])
    method2_result = method2_pearson_correlation(test_data, test_index, train_data, dynamic_top_k[1])
    method3_result = method3_pearson_correlation_with_IUF(test_data, test_index, train_data, dynamic_top_k[2], IUF)
    method4_result = method4_pearson_correlation_amplification(test_data, test_index, train_data, dynamic_top_k[3])
    method5_result = method5_pearson_correlation_with_IUF_AMP(test_data, test_index, train_data, dynamic_top_k[4], IUF)
    method6_result = method6_item_based_collaborative_filtering(test_data, test_index, train_data)
    rating_arr = [method1_result, method2_result, method3_result, method4_result, method5_result, method6_result]

    shape = rating_arr[0].shape
    if not all(rat.shape == shape for rat in rating_arr):
        print(f"Shape mismatch!❌")

    def majority_vote(values):
        count = Counter(values)
        most_common, freq = count.most_common(1)[0]
        return most_common

    all_rating_arr = np.array(rating_arr)

    valid_positions = np.all(all_rating_arr != -1, axis=0)
    result_matrix = np.full(shape, -1, dtype=int)

    majority_matrix = np.apply_along_axis(majority_vote, axis=0, arr=all_rating_arr)
    result_matrix[valid_positions] = majority_matrix[valid_positions]
    return result_matrix

# 0.7274253105068815 [0.4, 0.23, 0.21, 0.03, 0.01, 0.12]
# Best Weights: [0.45576116 0.06685901 0.08839644 0.08955054 0.02957128 0.26986157]
# Best MAE: 0.7244041624706278
def method8_weighted_ratings(test_data, test_index, test_avg, train_data, dynamic_top_k, IUF, weighted):
    method1_result = method1_cosine(test_data, test_index, test_avg, train_data, dynamic_top_k[0])
    method2_result = method2_pearson_correlation(test_data, test_index, train_data, dynamic_top_k[1])
    method3_result = method3_pearson_correlation_with_IUF(test_data, test_index, train_data, dynamic_top_k[2], IUF)
    method4_result = method4_pearson_correlation_amplification(test_data, test_index, train_data, dynamic_top_k[3])
    method5_result = method5_pearson_correlation_with_IUF_AMP(test_data, test_index, train_data, dynamic_top_k[4], IUF)
    method6_result = method6_item_based_collaborative_filtering(test_data, test_index, train_data)
    rating_arr = [method1_result, method2_result, method3_result, method4_result, method5_result, method6_result]
    shape = rating_arr[0].shape
    if not all(rat.shape == shape for rat in rating_arr):
        print(f"Shape mismatch!❌")

    total_rating = np.array(rating_arr)
    weights = weighted.reshape(-1, 1, 1)

    valid_positions = np.all(total_rating > 0, axis=0)
    result_matrix = np.full(shape, -1, dtype=int)

    weighted_sum = np.sum(total_rating * weights, axis=0)
    total_weight = np.sum(weights)
    weighted_avg_matrix = np.round(weighted_sum / total_weight).astype(int)

    result_matrix[valid_positions] = weighted_avg_matrix[valid_positions]
    return result_matrix

# Read file from train, test5,10,20
def parse_data_file(file, shape):
    data = np.full(shape, -1)
    file_index, file_no = extract_file_index(os.path.basename(file))
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                each_rate = line.split(" ")
                each_rate = [int(s.strip()) for s in each_rate]
                x = each_rate[0] - 1 - file_index
                y = each_rate[1] - 1
                data[x,y] = each_rate[2]
    return data, file_index

# get test index: 5:200, 10:300, 20:400, others: 160
def extract_file_index(file_name):
    file_index = 0
    file_no = 0
    if "5" in file_name:
        file_index = 200
        file_no = 5
    elif "10" in file_name:
        file_index = 300
        file_no = 10
    elif "20" in file_name:
        file_index = 400
        file_no = 20
    elif train_split_ground_truth == file_name or train_split_to_test == file_name or train_split_to_test_mix == file_name\
            or train_split_to_test_3 == file_name:
        file_index = 160
        file_no = 0
    return file_index, file_no

# Map the original unobserved data with predicted data
def extract_zero_indices(original_data):
    zero_dict = {}
    for i in range(original_data.shape[0]):
        row = original_data[i]
        zero_indices = np.where(row == 0)[0]
        zero_dict[i] = zero_indices
        # zero_dict[i] = [{j:to_extract_data[i][j]} for j in zero_indices]
    return zero_dict

def write_result_file(predicted_data, file, zero_dict_indices, file_suffix):
    print("💡Start writing result file ...")
    zero_dict = {}
    # {0:[0, 110, 282, ...], 1:[...], ...}
    for i, j in zero_dict_indices.items():
        zero_dict[i] = [{index:predicted_data[i][index]} for index in j]
    file_index, file_no = extract_file_index(os.path.basename(file))
    if os.path.basename(file) == "train.txt":
        file_index = 160
    file_name = RESULT_FILE_PATH + "result_of_" + str(file_no) + "_" + file_suffix + ".txt"
    with open(file_name, 'w') as f:
        for key, value in zero_dict.items():
            # users
            u = file_index + key + 1
            # movies
            m = 0
            r = 0
            for movie in value:
                m = list(movie.keys())[0] + 1
                # ratings
                r = list(movie.values())[0]
                f.write(f"{u} {m} {r}\n")
    print("✅Successfully writing result file ...")

def write_train_split_file(data, prefix, file_name):
    with open(file_name, 'w') as f:
        for i in range(data.shape[0]):
            each_train = data[i]
            for j in range(data.shape[1]):
                if each_train[j] != -1:
                    f.write(f"{prefix+i+1} {j+1} {each_train[j]}\n")

def preprocess_train_data(train_data_2):
    for i in range(train_data_2.shape[0]):
        each_train = train_data_2[i]
        observed_indices = np.where(each_train > 0)[0]
        # randomly select 5 indices from observed_indices
        # if len(observed_indices) > 20:
        #     selected_indices = random.sample(list(observed_indices), 20)
        # else:
        #     selected_indices = random.sample(list(observed_indices), 10)

        selected_indices = random.sample(list(observed_indices), 5)
        # set all others to 0
        for j in observed_indices:
            if j not in selected_indices:
                each_train[j] = 0
    return train_data_2

# momentum, velocity => not used anymore
def find_best_weight(MSE_func, init_weighted, max_iteration, learning_rate, tolerance, patient, momentum,
                    test_data, test_index, test_avg, train_data, dynamic_top_k, IUF, ground_truth_data):
    predicted_to_test = np.zeros_like(test_data)
    weight_ndarr = np.array(init_weighted)
    weight_ndarr /= np.sum(weight_ndarr)
    velocity = np.zeros_like(weight_ndarr)
    best_mse = float('inf')
    not_improved_counter = 0
    for each_iteration in range(max_iteration):
        gradient = np.zeros_like(weight_ndarr)
        predicted_to_test = method8_weighted_ratings(test_data, test_index, test_avg, train_data, dynamic_top_k, IUF, weight_ndarr)

        mask = (ground_truth_data != -1)
        error = predicted_to_test[mask] - ground_truth_data[mask]

        gradient = (2 / np.sum(mask)) * np.sum(error[:, None] * predicted_to_test[mask][:, None], axis=0)
        # update
        weight_ndarr -= learning_rate * gradient
        # velocity = momentum* velocity - learning_rate * gradient
        # weight_ndarr += velocity

        weight_ndarr = np.clip(weight_ndarr, 0, 1)
        weight_ndarr /= np.sum(weight_ndarr)

        curr_mse = MSE_func(predicted_to_test, ground_truth_data)
        if abs(best_mse - curr_mse) < tolerance:
            not_improved_counter += 1
        else:
            not_improved_counter = 0

        if curr_mse < best_mse:
            best_mse = curr_mse

        if not_improved_counter >= patient:
            print(f"❗️️️❗️Early stopping: No improvement in MSE at {each_iteration} iterations❗️❗️")
            print(f"Best Weights: {weight_ndarr}")
            print(f"Best MSE: {best_mse}")
            break

    return weight_ndarr, best_mse, predicted_to_test

def random_search(max_iteration, test_data, test_index, test_avg, train_data, dynamic_top_k, IUF, ground_truth_data):
    # best_mae = 0.7217186975495132
    best_mae = 0.7576367908694193
    best_weight_rand = None
    for _ in range(max_iteration):  # 200 - 300 iterations
        weight_random_arr = np.random.rand(6)
        weight_random_arr /= np.sum(weight_random_arr)  # make sure the sum is 1
        mae = wrap_MAE(test_data, test_index, test_avg, train_data, dynamic_top_k, IUF, weight_random_arr, ground_truth_data)

        if mae < best_mae:
            best_mae = mae
            best_weight_rand = weight_random_arr
    return best_mae, best_weight_rand

def wrap_MAE(test_data, test_index, test_avg, train_data, dynamic_top_k, IUF, weighted, ground_truth_data):
    if np.sum(weighted) != 1:
        # weights not sums to 1
        weighted = weighted / np.sum(weighted)
    predicted_to_test = method8_weighted_ratings(test_data, test_index, test_avg, train_data, dynamic_top_k, IUF, weighted)
    return MAE(predicted_to_test, ground_truth_data)

def MSE(predicted_data, original_data):
    mask = (original_data != -1)
    return np.mean((predicted_data[mask] - original_data[mask]) ** 2)

def MAE(predicted_data, original_data):
    mask = (original_data != -1)  # ground truth mask
    mae = np.sum(np.abs(predicted_data[mask] - original_data[mask])) / np.sum(mask)
    return mae

def split_train_data(train_data):
    # split train data into 80% and 20%
    train_data_1 = train_data[:160, :].copy()  # (k_160, 1000)
    train_data_2 = train_data[160:, :].copy()  # (40, 1000) serve as ground truth(starting from user 161)
    write_train_split_file(train_data_1, 0, "train_split_1.txt")
    write_train_split_file(train_data_2, 160, "train_split_ground_truth.txt")
    to_test = preprocess_train_data(train_data_2).copy()  # erase each train data to only 5/10/20 ratings like test data
    write_train_split_file(to_test, 160, "train_split_to_test_3.txt")

if __name__ == '__main__':
    train = "train.txt"
    test5 = "test5.txt"
    test10 ="test10.txt"
    test20 = "test20.txt"

    # train
    train_data, train_index = parse_data_file(train, (200,1000))

    # test
    test5_data, test5_index = parse_data_file(test5, (100,1000))
    zero_dict_5 = extract_zero_indices(test5_data)

    test10_data, test10_index = parse_data_file(test10, (100, 1000))
    zero_dict_10 = extract_zero_indices(test10_data)

    test20_data, test20_index = parse_data_file(test20, (100, 1000))
    zero_dict_20 = extract_zero_indices(test20_data)

    ##################################################################################################
    ################################## Split Train Data to test ######################################
    ##################################################################################################

    # generate split train data: 160:40 and resemble test5, 10, 20
    # split_train_data(train_data)

    # Predict train data 0️⃣
    # train_split_1_data, train_split_1_index = parse_data_file(train_split_1, (160, 1000))
    # # # print(f"train_split_1_data: \n{train_split_1_data}\n train_split_1_index: {train_split_1_index}")

    # train_split_ground_truth_data, train_split_ground_truth_index = parse_data_file(train_split_ground_truth, (40, 1000))
    # # # print(f"train_split_ground_truth_data: \n{train_split_ground_truth_data}\n train_split_ground_truth_index: {train_split_ground_truth_index}")

    # train_split_to_test_data, train_split_to_test_index = parse_data_file(train_split_to_test_3, (40, 1000))
    # # train_split_to_test_data, train_split_to_test_index = parse_data_file(train_split_to_test_mix, (40, 1000))
    # # # print(f"train_split_to_test_data: \n{train_split_to_test_data}\n train_split_to_test_index: {train_split_to_test_index}")

    # zero_dict_to_test = extract_zero_indices(train_split_to_test_data)
    # to_test_avg = compute_avg(train_split_to_test_data)
    # ground_truth_avg = compute_avg(train_split_ground_truth_data)

    # {k_160✅:0.815038603558241} {140✅:0.815038603558241} {120✅:0.815038603558241} {60:0.9056730446458543}
    # {100: 0.8915743538100034} {80:0.8939241356159785} {50:0.9244712990936556} {30: 0.9523329976502182} {15:0.9583752937227258}
    # {10:0.9013091641490433}{5:0.9013091641490433} {3:0.9137294394091977} {1:0.8919100369251427}
    # predicted_to_test = method1_cosine(train_split_to_test_data, 160, to_test_avg, train_split_1_data, 120)

    # {k_160✅:0.819402484055052} {140:0. 0.819402484055052} {130: 0.819402484055052} {110:0.8200738502853306} {60:0.8734474655924807}
    # {100: 0.8616145615010918} {50:0.8893065454372535} {20:  0.9456632169723044} {15:0.8459214501510574}
    # {10: 0.9311849613964418} {5:0.9318563276267203} {3:0.908694192682108} {1:0.8811681772406847}
    # predicted_to_test = method2_pearson_correlation(train_split_to_test_data, 160, train_split_1_data, 15)

    # {k_160✅:0.8153742866733803} {120:0.8153742866733803} {60:0.9472977509231285} {k_30:0.94662638469285}
    # {25:0.8472641826116146} {20:0.8506210137630077} {15:0.8647197045988587} {10:0.8915743538100034}
    # {5:0.8667338032896945} {3:0.8425646189996643} {1:0.8580060422960725}
    # IUF = compute_IUF(train_split_1_data)
    # predicted_to_test = method3_pearson_correlation_with_IUF(train_split_to_test_data, 160,
    #                                                         train_split_1_data, 120, IUF)

    # {k_160:0.8241020476670023✅} {140: 0.8294729775092313} {120: 0.8583417254112118} {100:0.8798254447801276}
    # {60:0.8892245720040282} {40:0.8905673044645854} {20:0.8630412890231621} {15: 0.9026518966096005}
    # {10:0.8479355488418933} {5:0.8382007385028533} {3:0.8234306814367237} {1:0.8811681772406847}
    # predicted_to_test = method4_pearson_correlation_amplification(train_split_to_test_data, 160,
    #                                                             train_split_1_data, 10)

    # {k_160✅:0.8261161463578383} {120:0.8261161463578383} {60:0.94662638469285} {k_30:0.9462907015777107}
    # {25:0.9365558912386707} {20:0.9352131587781135} {15:0.9288351795904666} {10:0.9003021148036254}
    # {5:0.9167505874454515} {3:0.9117153407183619} {1:0.8580060422960725}
    # IUF = compute_IUF(train_split_1_data)
    # predicted_to_test = method5_pearson_correlation_with_IUF_AMP(train_split_to_test_data,
    #                                                             160, train_split_1_data, 5, IUF)

    # 0.8210808996307486
    # predicted_to_test = method6_item_based_collaborative_filtering(train_split_to_test_data,
    #                                                             160, train_split_1_data)

    # 0.815038603558241
    # dynamic_top_k = [160,160,160,160,160]

    # terrible => stop using ❌
    # train_data = predict_train_data(train_data, 100, COSINE_SIMILAR)

    # terrible => stop using ❌
    # predicted_to_test = method7_majority_voting(train_split_to_test_data,
    #                                             160,
    #                                             to_test_avg,
    #                                             train_split_1_data,
    #                                             dynamic_top_k,
    #                                             IUF)

    ##################################################################################################
    ################################ Finding Best weight for method8 #################################
    ##################################################################################################

    # best_all_weight = None
    # best_all_MSE = float('inf')
    # predicted_to_test = np.zeros_like(train_split_to_test_data)
    # for i in range(10):
    #     # init_weight = [0.46090605, 0.05087891, 0.08082935, 0.07010856, 0.02877105, 0.30850609]
    #     # init_weight = [0.28747162, 0.00520519, 0.13519684, 0.05760594, 0.26752535, 0.24699507]
    #     init_weight = [0.45576116, 0.06685901, 0.08839644, 0.08155054, 0.02857128, 0.27886157]
    #     init_weight /= np.sum(init_weight)
          # ✅ 2️⃣: Refine the random search with Gradient Descent ✅
    #     best_weight, best_MSE, predicted_to_test = find_best_weight(MSE, init_weight, 200, 0.00001, 1e-6, 10, 0.9
    #                                             , train_split_to_test_data, 160, to_test_avg
    #                                             , train_split_1_data, dynamic_top_k, IUF, train_split_ground_truth_data)
    #     print(f"The {i}th time: Best Weights: {best_weight} Best MSE: {best_MSE}")
    #     if best_MSE < best_all_MSE:
    #         best_all_weight = best_weight
    #         best_all_MSE = best_MSE
    # print(f"Overall Best Weights: {best_all_weight}")
    # print(f"Overall Best MSE: {best_all_MSE}")


    # Best MAE: 0.5656260490097348
    # test10 & 20 Best weight: [0.28747162 0.00520519 0.13519684 0.05760594 0.26752535 0.24699507]
    # test10 & 20 Best weight: [0.28791626 0.0046109  0.13508101 0.05720452 0.26789658 0.24729073]
    # weighted = [0.28747162, 0.00520519, 0.13519684, 0.05760594, 0.26752535, 0.24699507]
    # weighted = np.array(weighted)
    # ✅ 1️⃣ : Use random search find a set of potential weights ✅
    # best_mae, best_weight_rand = random_search(200, train_split_to_test_data, 160, to_test_avg,
    #                                             train_split_1_data, dynamic_top_k, IUF, train_split_ground_truth_data)
    # print(f"Best MAE: {best_mae}")
    # print(f"Best Weights: {best_weight_rand}")

    # 0.7549513259483048
    # weighted = [0.45, 0.07, 0.09, 0.09, 0.03, 0.27]
    # test5: Best MAE: 0.7244041624706278
    # test5: weighted = [0.45576116, 0.06685901, 0.08839644, 0.08955054, 0.02957128, 0.26986157]
    # test5: Best MAE 0.7217186975495132
    # test5: Best weighted: [0.46090605, 0.05087891, 0.08082935, 0.07010856, 0.02877105, 0.30850609]

    # weighted = [0.46090605, 0.05087891, 0.08082935, 0.07010856, 0.02977105, 0.30750609]
    # weighted = [0.45576116, 0.06685901, 0.08839644, 0.08955054, 0.02957128, 0.26986157]
    # weighted = [0.45576116, 0.06685901, 0.08839644, 0.08155054, 0.02857128, 0.27886157]
    # weighted = np.array(weighted)
    # # 0.7549513259483048
    # ✅ 3️⃣ Apple Weighted Average => yield the best MAE ✅
    # predicted_to_test = method8_weighted_ratings(train_split_to_test_data,
    #                                             160,
    #                                             to_test_avg,
    #                                             train_split_1_data,
    #                                             dynamic_top_k,
    #                                             IUF,
    #                                             weighted)
    #
    # write_result_file(predicted_to_test, train, zero_dict_to_test, "method_8")
    # MAE_result = MAE(predicted_to_test, train_split_ground_truth_data)
    # print(f"MAE result for is : {MAE_result}")

    ##################################################################################################
    ######################################## Real Test Data ##########################################
    ##################################################################################################
    #todo:
    # Remember to comment out the astype for test_data and open the round() method in each method before
    # testing individually. To test method 8, first commenting out round() method and opening the
    # next one, then using weighted_20 for test20 and weighted_5_10 for test5 and 10.
    # To replicate the same result, use the same dynamic_top_k = [160, 160, 160, 160, 160].
    # To get the theoretically best result, use the dynamic_top_k = [200, 200, 200, 200, 200].

    # method 1️⃣ cosine
    # top_k = 160
    # test5_avg = compute_avg(test5_data)
    # predicted_result_5 = method1_cosine(test5_data, test5_index, test5_avg, train_data, top_k)
    # write_result_file(predicted_result_5, test5, zero_dict_5, "method_1_cosine")
    #
    # test10_avg = compute_avg(test10_data)
    # predicted_result_10 = method1_cosine(test10_data, test10_index, test10_avg, train_data, top_k)
    # write_result_file(predicted_result_10, test10, zero_dict_10, "method_1_cosine")
    #
    # test20_avg = compute_avg(test20_data)
    # predicted_result_20 = method1_cosine(test20_data, test20_index, test20_avg, train_data, top_k)
    # write_result_file(predicted_result_20, test20, zero_dict_20, "method_1_cosine")

    # method 2️⃣ Basic pearson
    # top_k = 160
    # predicted_result_5 = method2_pearson_correlation(test5_data, test5_index, train_data, top_k)
    # write_result_file(predicted_result_5, test5, zero_dict_5, "method_2_pearson_correlation")
    #
    # predicted_result_10 = method2_pearson_correlation(test10_data, test10_index, train_data, top_k)
    # write_result_file(predicted_result_10, test10, zero_dict_10, "method_2_pearson_correlation")
    #
    # predicted_result_20 = method2_pearson_correlation(test20_data, test20_index, train_data, top_k)
    # write_result_file(predicted_result_20, test20, zero_dict_20, "method_2_pearson_correlation")

    # method 3️⃣ pearson with IUF => The problem is that there is a risk of over-reliance on rare items
    # top_k = 160
    # IUF = compute_IUF(train_data)
    # predicted_result_5 = method3_pearson_correlation_with_IUF(test5_data, test5_index, train_data, top_k, IUF)
    # write_result_file(predicted_result_5, test5, zero_dict_5, "method_3_pearson_cor_IUF")
    #
    # predicted_result_10 = method3_pearson_correlation_with_IUF(test10_data, test10_index, train_data, top_k, IUF)
    # write_result_file(predicted_result_10, test10, zero_dict_10, "method_3_pearson_cor_IUF")
    #
    # predicted_result_20 = method3_pearson_correlation_with_IUF(test20_data, test20_index, train_data, top_k, IUF)
    # write_result_file(predicted_result_20, test20, zero_dict_20, "method_3_pearson_cor_IUF")

    # method 4️⃣ pearson with amplification
    # top_k = 160
    # predicted_result_5 = method4_pearson_correlation_amplification(test5_data, test5_index, train_data, top_k)
    # write_result_file(predicted_result_5, test5, zero_dict_5, "method_4_pearson_correlation_amp")
    #
    # predicted_result_10 = method4_pearson_correlation_amplification(test10_data, test10_index, train_data, top_k)
    # write_result_file(predicted_result_10, test10, zero_dict_10, "method_4_pearson_correlation_amp")
    #
    # predicted_result_20 = method4_pearson_correlation_amplification(test20_data, test20_index, train_data, top_k)
    # write_result_file(predicted_result_20, test20, zero_dict_20, "method_4_pearson_correlation_amp")

    # method 5️⃣ pearson with amplification and IUF
    # top_k = 160
    # IUF = compute_IUF(train_data)
    # predicted_result_5 = method5_pearson_correlation_with_IUF_AMP(test5_data, test5_index, train_data, top_k, IUF)
    # write_result_file(predicted_result_5, test5, zero_dict_5, "method_5_pearson_cor_IUF")
    #
    # predicted_result_10 = method5_pearson_correlation_with_IUF_AMP(test10_data, test10_index, train_data, top_k, IUF)
    # write_result_file(predicted_result_10, test10, zero_dict_10, "method_5_pearson_cor_IUF")
    #
    # predicted_result_20 = method5_pearson_correlation_with_IUF_AMP(test20_data, test20_index, train_data, top_k, IUF)
    # write_result_file(predicted_result_20, test20, zero_dict_20, "method_5_pearson_cor_IUF")

    # method 6️⃣ item-based collaborative filtering
    # predicted_result_5 = method6_item_based_collaborative_filtering(test5_data, test5_index, train_data)
    # write_result_file(predicted_result_5, test5, zero_dict_5, "method_6_item_based_cf_adjusted_cosine")
    #
    # predicted_result_10 = method6_item_based_collaborative_filtering(test10_data, test10_index, train_data)
    # write_result_file(predicted_result_10, test10, zero_dict_10, "method_6_item_based_cf_adjusted_cosine")
    #
    # predicted_result_20 = method6_item_based_collaborative_filtering(test20_data, test20_index, train_data)
    # write_result_file(predicted_result_20, test20, zero_dict_20, "method_6_item_based_cf_adjusted_cosine")

    # method 7️⃣ majority_voting method
    # dynamic_top_k = [160,160,160,160,160]
    # test5_avg = compute_avg(test5_data)
    # test10_avg = compute_avg(test10_data)
    # test20_avg = compute_avg(test20_data)
    # IUF = compute_IUF(train_data)
    # predicted_result_5 = method7_majority_voting(test5_data, test5_index, test5_avg, train_data, dynamic_top_k, IUF)
    # write_result_file(predicted_result_5, test5, zero_dict_5, "method_7_self_defined_method")
    #
    # predicted_result_10 = method7_majority_voting(test10_data, test10_index, test10_avg, train_data, dynamic_top_k, IUF)
    # write_result_file(predicted_result_10, test10, zero_dict_10, "method_7_self_defined_method")
    #
    # predicted_result_20 = method7_majority_voting(test20_data, test20_index, test20_avg, train_data, dynamic_top_k, IUF)
    # write_result_file(predicted_result_20, test20, zero_dict_20, "method_7_self_defined_method")

    # method 8️⃣ weighted ratings
    # dynamic_top_k = [160, 160, 160, 160, 160] # to replicate the same result
    # dynamic_top_k = [200, 200, 200, 200, 200] # theoretically better
    # weighted_20 = [0.45576116, 0.06685901, 0.08839644, 0.08955054, 0.02957128, 0.26986157] # for test20
    # weighted_20 = np.array(weighted_20)
    # weighted_5_10 = [0.46090605, 0.05087891, 0.08082935, 0.07010856, 0.02977105, 0.30750609] # for test5, 10
    # weighted_5_10 = np.array(weighted_5_10)
    # test5_avg = compute_avg(test5_data)
    # test10_avg = compute_avg(test10_data)
    # test20_avg = compute_avg(test20_data)
    # IUF = compute_IUF(train_data)
    # predicted_result_5 = method8_weighted_ratings(test5_data, test5_index, test5_avg, train_data, dynamic_top_k, IUF, weighted)
    # write_result_file(predicted_result_5, test5, zero_dict_5, "method_8_weighted_ratings")
    # predicted_result_10 = method8_weighted_ratings(test10_data, test10_index, test10_avg, train_data, dynamic_top_k, IUF, weighted)
    # write_result_file(predicted_result_10, test10, zero_dict_10, "method_8_weighted_ratings")
    # predicted_result_20 = method8_weighted_ratings(test20_data, test20_index, test20_avg, train_data, dynamic_top_k, IUF, weighted)
    # write_result_file(predicted_result_20, test20, zero_dict_20, "method_8_weighted_ratings")