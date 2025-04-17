import csv
import numpy
import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from symspellpy import SymSpell, Verbosity
import nltk

nltk.download('punkt_tab')
# init SymSpell
symSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
symSpell.load_dictionary("./frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

def correct_mispronounciation(text):
    possible_answer = symSpell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)
    if possible_answer:
        return possible_answer[0].term
    return text

def import_csv(file_path, is_label=False):
    document = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        stemmer = PorterStemmer()
        for each_line in reader:
            if len(each_line) > 0:
                if is_label:
                    label = int(each_line[0].strip())
                    document.append(label)
                else:
                    # text = each_line[0].strip() + " " # append brand name to text
                    text = each_line[1].strip()
                    text = text.lower() # lower case
                    text = re.sub(r"@\w+", '', text) # remove @
                    text = re.sub(r"#\w+", '', text) # remove #
                    text = re.sub(r'\d+', '', text) # remove numbers
                    text = re.sub(r'[^a-zA-Z0-9\s]', " ", text) # remove special characters
                    text = re.sub(r'\s+', " ", text).strip() # remove extra spaces
                    tokens = word_tokenize(text)
                    # correct mispronounciation
                    corrected_tokens = [correct_mispronounciation(word) for word in tokens]
                    # word stemming
                    after_stem = [stemmer.stem(each_word) for each_word in corrected_tokens]
                    cleaner_text = " ".join(after_stem)
                    # print(cleaner_text)
                    document.append(cleaner_text)
    return document

def import_weight_csv(file_path):
    weight = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for each_line in reader:
            for each_weight in each_line:
                # print(each_weight[0])
                weight.append(float(each_weight.strip('[]')))
    return weight

def convert_data_to_tfidf(doc, vectorizer, is_test=False):
    if not is_test:
        # Fit and transform the doc into TF-IDF
        tfidf_matrix = vectorizer.fit_transform(doc)
        tfidf_np = numpy.array(tfidf_matrix.toarray())
        print(f"Initial tf-idf shape: {tfidf_np.shape}")

        # Appending 1 at the end becomes X_train
        tfidf_np = numpy.array(tfidf_matrix.toarray())
        bias = np.ones((tfidf_np.shape[0], 1))  # Create a column of ones
        X_train = np.hstack([tfidf_np, bias])
        # print(f"Last column is bias: {X_train[:, -1]}")
        # print(f"After appending 1 at the end: {X_train.shape}")
        # print(X_train)
        return X_train
    else:
        test_tfidf_matrix = vectorizer.transform(doc)
        test_tfidf_np = numpy.array(test_tfidf_matrix.toarray())
        bias = np.ones((test_tfidf_np.shape[0], 1))
        X_test = np.hstack([test_tfidf_np, bias])
        return X_test

def sigmoid(r):
    return 1 / (1 + np.exp(-r))

def monitor_loss(y_train, y_predict):
    epsilon = 1e-15
    y_predict = np.clip(y_predict, epsilon, 1-epsilon)
    loss = -np.mean(y_train * np.log(y_predict) + (1 - y_train) * np.log(1 - y_predict))
    return loss

def monitor_data_likelihood(y_train, y_predict):
    epsilon = 1e-15
    y_predict = np.clip(y_predict, epsilon, 1-epsilon)
    return np.mean(y_train * np.log(y_predict) + (1 - y_train) * np.log(1 - y_predict))

# gradient ascent but monitor cross entropy loss & data likelihood during training
def fit_gd(X_train, y_train, eta=0.01, n_iters=30000, epsilon=1e-6):
    print("Start fitting...")
    weight = np.zeros((X_train.shape[1], 1))
    cur_iter = 1
    prev_loss = float('inf')
    while cur_iter < n_iters:
        y_pred = sigmoid(X_train.dot(weight))
        gradient = X_train.T.dot(y_train - y_pred)
        weight += eta * gradient

        cur_loss = monitor_loss(y_train, y_pred)
        if cur_iter % 10000 == 0 and cur_iter > 0:
            data_likelihood = monitor_data_likelihood(y_train, y_pred)
            print(f"Iteration:{cur_iter} Loss: {cur_loss:.4f} Data Likelihood: {data_likelihood:.4f}")
            eta = eta * 0.9
            print(f"New eta: {eta:.6f}")
        if abs(prev_loss - cur_loss) < epsilon:
            print(f"Early Break at iteration {cur_iter} Current Loss: {cur_loss:.6f} Previous Loss: {prev_loss:.6f}")
            break
        if cur_loss < prev_loss:
            prev_loss = cur_loss
        cur_iter += 1
    return weight

def predict(X_test, weight, threshold=0.5):
    s = sigmoid(X_test.dot(weight)) # shape: (n,1)
    y_predict = np.where(s >= threshold, 1, 0) # if s > threshold, y_predict = 1, else y_predict = 0
    return y_predict

def MSE(y_predict, y_test):
    return np.mean((y_predict - y_test) ** 2)

def accuracy(y_predict, y_test):
    correct = np.sum(y_predict == y_test)
    total = len(y_test)
    return correct / total

def fit_and_transform(X_train, y_train, max_feature_cnt, split_ratio=0.8, total_round=1):
    # split the data
    to_split_index = int(len(X_train) * split_ratio)
    X_train_split = X_train[:to_split_index]
    y_train_split = y_train[:to_split_index]
    y_train_split = y_train_split.reshape(-1, 1)
    # mock test
    X_test_split = X_train[to_split_index:]
    y_test_split = y_train[to_split_index:]
    y_test_split = y_test_split.reshape(-1, 1)

    round_best_weight = np.zeros((X_train.shape[1], 1))
    round_best_mse = float('inf')
    for i in range(total_round):
        # print(f"Round {i}")
        train_weight = fit_gd(X_train_split, y_train_split)
        y_predict = predict(X_test_split, train_weight)
        mse = MSE(y_predict, y_test_split)
        # print("MSE: ", mse)
        if mse < round_best_mse:
            round_best_mse = mse
            round_best_weight = train_weight

    # print("Best MSE: ", round_best_mse)
    # write_weight(round_best_weight, "./weight_history", "round_best_weight", round_best_mse, max_feature_cnt)
    return round_best_mse, round_best_weight

def write_weight(weight, file_path, file_name, MSE_VAL, max_feature_cnt):
    file_path = f"{file_path}/{file_name}_{MSE_VAL:.4f}_{max_feature_cnt}.csv"
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        # print(f"weight shape: {weight.shape}")
        writer.writerow(weight.reshape(-1,1))

def output_test_file(x_test_document, ultra_weight, file_path, vectorizer):
    X_test = convert_data_to_tfidf(x_test_document, vectorizer, True)
    print(f"Test data shape: {X_test.shape}")
    s_val = sigmoid(X_test.dot(ultra_weight))
    np.savetxt(file_path, s_val, fmt='%.6f')
    print(f"Writing file successfully to {file_path} !!!")

if __name__ == '__main__':
    x_train_path = "./train_data/x_train.csv"
    y_train_path = "./train_data/y_train.csv"
    x_test_path = "./test_data/x_test.csv"

    x_document = import_csv(x_train_path) # len: 2400
    y_label = import_csv(y_train_path, True) # len: 2400

    x_test_doc = import_csv(x_test_path) # len: 600


    # with ngram_range=(1,2), max_feature: 1000, sublinear_tf= False or True => accuracy: 0.91875
    # with ngram_range=(1,2), max_feature: 1100, sublinear_tf= True => accuracy: 0.9191666666666667
    # with ngram_range=(1,2), max_feature: 1200, sublinear_tf= True => accuracy: 0.9229166666666667
    # ft_cnt = None # 0.90375
    ft_cnt = 1200
    # best_training_accuracy = 0.9229166666666667
    # best_max_ft = -1
    # for i in range(1200,1400,200):
        # print(f"max_features: {i}")
    vectorizer = TfidfVectorizer(max_features=ft_cnt, ngram_range=(1,2), sublinear_tf=True)
    X_train = convert_data_to_tfidf(x_document, vectorizer)
    y_train = numpy.array(y_label).reshape(-1, 1)

    # first inner test by splitting the data
    best_mse, best_weight = fit_and_transform(X_train, y_train, ft_cnt)
    train_pred = predict(X_train, best_weight)
    cur_accuracy = accuracy(train_pred, y_train)
    print(f"Training Accuracy: {cur_accuracy}")
    print(f"============= Mock training over =============")
        # if cur_accuracy > best_training_accuracy:
        #     best_training_accuracy = cur_accuracy
        #     best_max_ft = i
    # print(f"Best Training Accuracy: {best_training_accuracy}")
    # print(f"Best max_features: {best_max_ft}")

    # write weight to file
    # write_weight(best_weight, "./weight_history", "finalV2_ultra_best_weight", best_mse, ft_cnt)

    # 🗂️ write test file "yprob_test.txt"
    # print("Start training all X_train & y_train ...")
    # total_train_weight = fit_gd(X_train, y_train)
    # file_path_for_test = "predict_test/yprob_test.txt"
    # output_test_file(x_test_doc, total_train_weight, file_path_for_test, vectorizer)
