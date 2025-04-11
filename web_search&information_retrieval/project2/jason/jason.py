import numpy as np
import math

training_matrix = np.zeros((200, 1000))

class UMRList:
    def __init__(self, user_list, movie_list, rating_list,):
        self.user_list = user_list
        self.movie_list = movie_list
        self.rating_list = rating_list

    def get_user_list(self):
        return self.user_list

    def get_movie_list(self):
        return self.movie_list

    def get_rating_list(self):
        return self.rating_list

    def set_user_list(self, user_list):
        self.user_list = user_list

    def set_movie_list(self, movie_list):
        self.movie_list = movie_list

    def set_rating_list(self, rating_list):
        self.rating_list = rating_list


class MatrixUserAvgRating:
    def __init__(self, user_id, avg_rating):
        self.user_ID = user_id
        self.avg_rating = avg_rating

    def get_user_id(self):
        return self.user_ID

    def get_avg_rating(self):
        return self.avg_rating

    def set_usr_id(self, user_id):
        self.user_ID = user_id

    def set_avg_rating(self, avg_rating):
        self.avg_rating = avg_rating


class KNeighbor:
    def __init__(self, user_id, similarity):
        self.user_id = user_id
        self.similarity = similarity

    def get_similarity(self):
        return self.similarity

    def get_user_id(self):
        return self.user_id


def load_training_matrix():
    # build up a training matrix
    training_file = open('train.txt', 'r')
    matrix = np.zeros((200, 1000))# user 200 movie 1000
    line = training_file.readline()
    while line != '':
        line = line.split()
        each_value = [int(j) for j in line]
        userid = each_value[0] - 1
        movie = each_value[1] - 1
        rating = each_value[2]
        matrix[userid][movie] = rating
        line = training_file.readline()

    training_file.close()
    return matrix

# extract prediction movie list
def prediction_movie_list():
    movie_list = []
    rating_list = []
    user_list = []

    test_file = open('test5.txt', 'r')
    line = test_file.readline()

    while line:
        line = line.split()
        value = []
        for i in line:
            value.append(int(i))

        if value[2] == 0:
            user_list.append(value[0])
            movie_list.append(value[1])
            rating_list.append(value[2])

        line = test_file.readline()

    # exact prediction data: usr, mv, rating and store to a list
    prediction_list = UMRList(user_list, movie_list, rating_list)

    test_file.close()
    return prediction_list


# not finished yet
def avg_rating_from_training():
    train_rating_list = {}

    for usr in range(0, 200):
        non_rate_count = 0
        sum_rating = 0.0

        for movie in range(0, 1000):
            if training_matrix[usr][movie] != 0:
                sum_rating += training_matrix[usr][movie]*1.0
                non_rate_count += 1

        avg_rating = sum_rating/non_rate_count
        # index is user ID
        train_rating_list[usr+1] = avg_rating

    return train_rating_list
# not finished yet


# calculate average rating for each movie in training matrix
def each_movie_avg_rating():
    each_movie_avg_rating_list = {}

    for movie in range(0, 1000):
        sum_rating = 0.0
        count = 0

        for usr in range(0, 200):
            if training_matrix[usr][movie] != 0:
                sum_rating += training_matrix[usr][movie]
                count += 1

        if count > 0:
            avg_rating = sum_rating/count
            each_movie_avg_rating_list[movie + 1] = avg_rating
        else:
            each_movie_avg_rating_list[movie + 1] = 0.0

    return each_movie_avg_rating_list


# extract given data in test file and cal average rating
def given_list_in_test(usr_id):
    user_list = []
    movie_list = []
    rating_list = []

    test = open('test5.txt', 'r')
    line = test.readline()

    while line:
        line = line.split()
        value = []
        for i in line:
            value.append(int(i))

        user = value[0]
        movie = value[1]
        rating = value[2]

        if (usr_id == user) & (rating != 0):
            user_list.append(user)
            movie_list.append(movie)
            rating_list.append(rating)

        line = test.readline()

    test.close()

    given_object = UMRList(user_list, movie_list, rating_list)
    return given_object


# calculate average given rating, in test5.txt, avg = sum/5
def avg_user_rating_in_test(usr_id):
    user_list = []
    movie_list = []
    rating_list = []

    test = open('test5.txt', 'r')
    line = test.readline()

    while line:
        line = line.split()
        value = []
        for i in line:
            value.append(int(i))

        user = value[0]
        movie = value[1]
        rating = value[2]

        if (user == usr_id) & (rating != 0):
            user_list.append(user)
            movie_list.append(movie)
            rating_list.append(rating)

        line = test.readline()

    test.close()

    # here is calculate avg given rating
    total_rating = 0.0
    size = len(movie_list)

    for i in range(0, size):
        total_rating += rating_list[i]

    avg_rating = total_rating/size
    return avg_rating


def cal_cosine_similarity(user_id, k_neighbor):
    # calculate cosine similarity, find nearest neighbor
    neighbor_list = []

    given_object = given_list_in_test(user_id)
    movie_list = given_object.get_movie_list()
    rating_list = given_object.get_rating_list()

    for usr in range(0, 200):
        numerator = 0.0
        denom_test = 0.0
        denom_train = 0.0
        common_movie = 0

        for movie in range(0, len(movie_list)):
            movie_id = movie_list[movie]
            # both rating in test and in train are not zero
            if (rating_list[movie] != 0) & (training_matrix[usr][movie_id-1] != 0):
                numerator += rating_list[movie] * training_matrix[usr][movie_id-1]

                denom_test += math.pow(rating_list[movie], 2)
                denom_train += math.pow(training_matrix[usr][movie_id-1], 2)
                common_movie += 1

        denom_total = math.sqrt(denom_test) * math.sqrt(denom_train)

        if denom_total != 0.0:
            similarity = numerator / denom_total
            neighbor_obj = KNeighbor(usr + 1, similarity)
            neighbor_list.append(neighbor_obj)

    # sort need top K nearest neighbor, para is K_neighbor
    neighbor_list.sort(key=lambda x: x.similarity, reverse=True)
    k_neighbor_obj_list = []

    for i in range(0, len(neighbor_list)):
        if i < k_neighbor:
            k_neighbor_obj_list.append(neighbor_list[i])

    # top k neighbors
    return k_neighbor_obj_list


# calculate movie rating with vector similarity
def cal_rating_cosine(user_id, movie_id, k_neighbor):
    similarity_list = cal_cosine_similarity(user_id, k_neighbor)

    # calculate the movie average rating
    avg_movie_rating = each_movie_avg_rating()
    the_avg_movie_rating = avg_movie_rating[movie_id]

    # calculate the user average rating in test file (five guys, yummy)
    avg_user_rating = avg_user_rating_in_test(user_id)

    numerator = 0.0
    denominator = 0.0

    for i in range(0, len(similarity_list)):
        neighbor = similarity_list[i]
        training_user_id = neighbor.get_user_id()

        if training_matrix[training_user_id-1][movie_id-1] > 0:
            rating = training_matrix[training_user_id-1][movie_id-1]

            numerator += neighbor.get_similarity() * rating
            denominator += neighbor.get_similarity()

    # if denominator is zero, then the prediction rating is average
    # of total movies in training matrix
    # else result is equal to average in test file
    if denominator != 0:
        result = numerator/denominator
    elif the_avg_movie_rating != 0:
        result = the_avg_movie_rating
    else:
        result = avg_user_rating

    result = int(round(result))
    # just in case
    if result > 5:
        result = 5
    elif result < 0:
        result = 1
        
    return result


def inverse_user_frequency(movie_id):
    # formula: IUF(j) = log(m/mj)
    # m is total number of users = 200
    m = 0
    for usr in range(0, 200):
        if training_matrix[usr][movie_id-1] != 0:
            m += 1
    if m != 0:
        iuf = math.log10(200.0/m)
    else:
        iuf = 1

    return iuf


def pearson_correlation(user_id, k_neighbor):
    # calculate pearson correlation find nearest neighbor
    neighbor_list = []

    avg_rating_in_test = avg_user_rating_in_test(user_id)

    given_object = given_list_in_test(user_id)
    movie_list = given_object.get_movie_list()
    rating_list = given_object.get_rating_list()

    # here return a list key is user id value is for this
    # user, average rating of 1000 movies in taring matrix
    train_rating_list = avg_rating_from_training()

    for usr in range(0, 200):
        numerator = 0.0
        denom_test = 0.0
        denom_train = 0.0
        common_movie = 0

        # iuf_mv_id = 0
        for movie in range(0, len(movie_list)):
            movie_id = movie_list[movie]
            # iuf_mv_id = movie_id
            usr_rating = rating_list[movie]
            train_rating = training_matrix[usr][movie_id-1]

            # both rating in test and in train are not zero
            if (usr_rating != 0) & (train_rating != 0):

                avg_rating_in_train = train_rating_list[usr+1]

                numerator += (usr_rating - avg_rating_in_test) * (train_rating - avg_rating_in_train)

                denom_test += math.pow((usr_rating - avg_rating_in_test), 2)
                denom_train += math.pow((train_rating - avg_rating_in_train), 2)
                common_movie += 1

        denom_total = 0.0
        if (denom_test > 0) & (denom_train > 0):
            denom_total = math.sqrt(denom_test) * math.sqrt(denom_train)

        if denom_total != 0.0:
            similarity = numerator/denom_total

            # my own way: Dirichlet smoothing
            # similarity *= (common_movie/(common_movie + 2))

            # improving prediction II: case amplification
            # similarity *= math.pow(math.fabs(similarity), 1.5)

            # improving prediction I: IUF
            # iuf = inverse_user_frequency(iuf_mv_id)

            # similarity *= iuf

            neighbor_obj = KNeighbor(usr+1, similarity)
            neighbor_list.append(neighbor_obj)

    # sort need top K nearest neighbor, para is K_neighbor
    neighbor_list.sort(key=lambda x: x.similarity, reverse=True)
    k_neighbor_obj_list = []

    # extract top k neighbors
    for i in range(0, len(neighbor_list)):
        if i < k_neighbor:
            k_neighbor_obj_list.append(neighbor_list[i])

    # top k neighbor-objects
    return k_neighbor_obj_list


# calculate pearson
def cal_rating_pearson(movie_id, similarity_list, avg_user_rating, the_avg_movie_rating, train_rating_list):
    numerator = 0.0
    denominator = 0.0

    for i in range(0, len(similarity_list)):
        neighbor = similarity_list[i]
        training_user_id = neighbor.get_user_id()

        # here return a list key is user id value is for this
        # user, average rating of 1000 movies in taring matrix
        avg_rating_in_train = train_rating_list[training_user_id]

        if training_matrix[training_user_id-1][movie_id-1] > 0:
            weight = training_matrix[training_user_id-1][movie_id-1] - avg_rating_in_train
            numerator += weight * neighbor.get_similarity()
            denominator += math.fabs(neighbor.get_similarity())

    # if denominator is zero, then the prediction rating is average
    # of total movies in training matrix
    # else result is equal to average in test file
    if denominator != 0:
        result = avg_user_rating + numerator/denominator
    elif the_avg_movie_rating != 0:
        result = the_avg_movie_rating
    else:
        result = avg_user_rating

    result = int(round(result))
    # just in case
    if result > 5:
        result = 5
    elif result < 0:
        result = 1.0

    return result

item_usr_mv_list = {}


def item_based_with_adjusted_cosine(user_id, movie_id, k_neighbor):
    # calculate adjusted cosine similarity, find nearest neighbor
    neighbor_list = []

    given_object = given_list_in_test(user_id)
    movie_list = given_object.get_movie_list()

    # here return a list key is user id value is for this
    # user, average rating of 1000 movies in taring matrix
    train_rating_list = avg_rating_from_training()

    for usr in range(len(movie_list)):
        rated_movie = movie_list[usr]
        numerator = 0.0
        denom_i = 0.0
        denom_j = 0.0

        common_usr = 0

        for user in range(0, 200):
            avg_rating_in_train = train_rating_list[user+1]
            if(training_matrix[user][rated_movie-1] != 0) & (training_matrix[user][movie_id-1] != 0):
                numerator += (training_matrix[user][rated_movie-1] -
                              avg_rating_in_train) * (training_matrix[user][movie_id-1] - avg_rating_in_train)
                denom_i += math.pow((training_matrix[user][rated_movie-1]), 2)
                denom_j += math.pow((training_matrix[user][movie_id-1]), 2)
                common_usr += 1

        if common_usr > 1:
            denom_total = math.sqrt(denom_i) * math.sqrt(denom_j)
            similarity = numerator/denom_total

            # Dirichlet smoothing
            similarity *= (common_usr/(common_usr + 2))

            # improving prediction II: case amplification
            similarity *= math.pow(math.fabs(similarity), 1.5)

            neighbor_obj = KNeighbor(usr+1, similarity)
            neighbor_list.append(neighbor_obj)
            item_usr_mv_list[usr+1] = movie_id

    # sort need top K nearest neighbor, para is K_neighbor
    neighbor_list.sort(key=lambda x: x.similarity, reverse=True)
    k_neighbor_obj_list = []

    for i in range(0, len(neighbor_list)):
        if i < k_neighbor:
            k_neighbor_obj_list.append(neighbor_list[i])

    # top k neighbors
    return k_neighbor_obj_list


def cal_rating_item_based(user_id, movie_id, k_neighbor):
    similarity_list = item_based_with_adjusted_cosine(user_id, movie_id, k_neighbor)

    # calculate the user average rating in test file (five guys, yummy)
    avg_user_rating = avg_user_rating_in_test(user_id)

    # calculate the movie average rating
    avg_movie_rating = each_movie_avg_rating()
    the_avg_movie_rating = avg_movie_rating[movie_id]

    numerator = 0.0
    denominator = 0.0

    given_object = given_list_in_test(user_id)
    movie_list = given_object.get_movie_list()
    rating_list = given_object.get_rating_list()

    for i in range(0, len(movie_list)):
        movie_id = movie_list[i]
        rating = rating_list[i]

        for j in range(0, len(similarity_list)):
            neighbor = similarity_list[j]
            similarity = neighbor.get_similarity()
            usr_id = neighbor.get_user_id()
            similar_mv_id = item_usr_mv_list[usr_id]

            if movie_id == similar_mv_id:
                numerator += similarity * rating
                denominator += math.fabs(similarity)

    if denominator != 0.0:
        result = numerator/denominator
    elif the_avg_movie_rating != 0:
        result = the_avg_movie_rating
    else:
        result = avg_user_rating

    result = int(round(result))

    # just in case
    if result > 5:
        result = 5
    elif result < 0:
        result = 1

    return result


# try to run the program
def execute():
    k_neighbor = 100
    output_list = prediction_movie_list()
    output_file = open('result5_pearson_basic.txt', 'w')

    # for all prediction users
    avg_movie_rating = each_movie_avg_rating()
    train_rating_list = avg_rating_from_training()
    last_user = -1
    similarity_list = []
    avg_user_rating = -1
    for i in range(0, len(output_list.get_user_list())):
        user_id = output_list.get_user_list()[i]
        movie_id = output_list.get_movie_list()[i]
        if last_user != user_id:
            similarity_list = pearson_correlation(user_id, k_neighbor)
            # calculate the user average rating in test file (five guys, yummy)
            avg_user_rating = avg_user_rating_in_test(user_id)

        # calculate the movie average rating
        the_avg_movie_rating = avg_movie_rating[movie_id]

        # cosine similarity
        # rating_cosine = cal_rating_cosine(user_id, movie_id, k_neighbor)
        # pearson correlation
        rating_pearson = cal_rating_pearson(movie_id, similarity_list,
                                            avg_user_rating, the_avg_movie_rating, train_rating_list)

        # own algorithm 0.6 * cosine + 0.4 * pearson
        # rating_final = int(round(0.6 * rating_cosine + 0.4 * rating_pearson))

        # item-based rating with adjusted cosine
        # rating_item_based = cal_rating_item_based(user_id, movie_id, k_neighbor)

        line = str(user_id) + " " + str(movie_id) + " " + str(rating_pearson)
        output_file.write(line + '\n')
        print(f"{i+1}/{len(output_list.get_user_list())} processed")
        last_user = user_id
    output_file.close()
    done = "Completed"
    return print(done)


if __name__ == '__main__':
    # initialization
    training_matrix = load_training_matrix()
    execute()