import csv
import numpy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def import_csv(file_path, is_label=False):
    document = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for each_line in reader:
            if len(each_line) > 0:
                if is_label:
                    label = int(each_line[0].strip())
                    document.append(label)
                else:
                    text = each_line[1].strip()
                    document.append(text)
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

def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

def monitor_loss(y_train, y_predict):
    epsilon = 1e-15
    if np.any(np.isnan(y_predict)):
        print("y_predict contains NaN values❗️")
    y_predict = np.clip(y_predict, epsilon, 1-epsilon)
    loss = -np.mean(y_train * np.log(y_predict) + (1 - y_train) * np.log(1 - y_predict))
    return loss

def monitor_data_likelihood(y_train, y_predict):
    return np.mean(y_train * np.log(y_predict) + (1 - y_train) * np.log(1 - y_predict))

# best 30000
def fit_gd(X_train, y_train, eta=0.01, n_iters=20000, epsilon=1e-6):
    print("Start fitting...")
    # weight = np.random.randn(X_train.shape[1], 1) * 0.01
    weight = np.zeros((X_train.shape[1], 1))
    cur_iter = 1
    prev_loss = float('inf')
    while cur_iter < n_iters:
        cur_iter += 1
        y_pred = sigmoid(X_train.dot(weight))
        gradient = X_train.T.dot(y_train - y_pred)
        weight += eta * gradient

        cur_loss = monitor_loss(y_train, y_pred)
        if cur_iter % 1000 == 0 and cur_iter > 0:
            data_likelihood = monitor_data_likelihood(y_train, y_pred)
            print(f"Iteration:{cur_iter} Loss: {cur_loss:.4f} Data Likelihood: {data_likelihood:.4f}")
            # eta /= 2
            # print(f"Current eta: {eta:.6f}")
        if abs(prev_loss - cur_loss) < epsilon:
            print(f"Break at iteration {cur_iter} Current Loss: {cur_loss:.6f} Previous Loss: {prev_loss:.6f}")
            break
        if cur_loss < prev_loss:
            prev_loss = cur_loss
    return weight

def fit_stochastic_gd(X_train, y_train, eta=0.1, n_iters=30000, epsilon=1e-5, batch_size=32):
    tolearance = 10
    weight = np.random.randn(X_train.shape[1], 1) * 0.01
    # weight = np.zeros((X_train.shape[1], 1))
    total_X_len = X_train.shape[0]
    prev_loss = float('inf')

    for epoch in range(n_iters):
        indices = np.random.permutation(total_X_len)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # predict for every batch
        for start in range(0, total_X_len, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            y_pred = sigmoid(X_batch.dot(weight))
            gradient = X_batch.T.dot((y_batch - y_pred))
            weight += eta * gradient / batch_size

        full_pred = sigmoid(X_train.dot(weight))
        cur_loss = monitor_loss(y_train, full_pred)
        if epoch % 1000 == 0 and epoch > 0:
            print(f"Epoch:{epoch} Loss: {cur_loss:.6f}")
            eta /= 2
            print(f"Current eta: {eta:.6f}")
        if abs(prev_loss - cur_loss) < epsilon:
            if tolearance == 0:
                print(f"Break at epoch {epoch} Current Loss: {cur_loss:.6f} Previous Loss: {prev_loss:.6f}")
                break
            tolearance -= 1
        if cur_loss < prev_loss:
            prev_loss = cur_loss

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

def inner_split_train_test(X_train, y_train, max_feature_cnt, split_ratio=0.8, total_round=1):
    # split the data
    to_split_index = int(len(X_train) * split_ratio)
    X_train_split = X_train[:to_split_index]
    y_train_split = y_train[:to_split_index]
    y_train_split = y_train_split.reshape(-1, 1)
    # mock test
    X_test_split = X_train[to_split_index:]
    y_test_split = y_train[to_split_index:]
    y_test_split = y_test_split.reshape(-1, 1)

    best_weight = np.zeros((X_train.shape[1], 1))
    best_mse = float('inf')
    best_accuracy = float('-inf')
    for i in range(total_round):
        print(f"Round {i}")
        train_weight = fit_gd(X_train_split, y_train_split)
        # train_weight = fit_stochastic_gd(X_train_split, y_train_split)
        y_predict = predict(X_test_split, train_weight)
        mse = MSE(y_predict, y_test_split)
        print("MSE: ", mse)
        if mse < best_mse:
            best_mse = mse
            best_weight = train_weight

    print("Best MSE: ", best_mse)
    # write_weight(best_weight, "./weight_history", "best_weight", best_mse, max_feature_cnt)
    return best_mse, best_weight

def write_weight(weight, file_path, file_name, MSE_VAL, max_feature_cnt):
    file_path = f"{file_path}/{file_name}_{MSE_VAL:.4f}_{max_feature_cnt}.csv"
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        # print(f"weight shape: {weight.shape}")
        writer.writerow(weight.reshape(-1,1))

def output_test_file(x_test_document, ultra_weight, file_path, vectorizer):
    X_test = convert_data_to_tfidf(x_test_document, vectorizer, True)
    s_val = sigmoid(X_test.dot(ultra_weight))
    np.savetxt(file_path, s_val, fmt='%.6f')

if __name__ == '__main__':
    x_train_path = "./train_data/x_train.csv"
    y_train_path = "./train_data/y_train.csv"
    x_test_path = "./test_data/x_test.csv"

    x_document = import_csv(x_train_path) # len: 2400
    y_label = import_csv(y_train_path, True) # len: 2400

    x_test_doc = import_csv(x_test_path) # len: 600

    # there are total 4510 features
    # best_mse_final = float('inf')
    # best_ft_cnt = 0
    # best_accuracy = float('-inf')
    # for ft_cnt in range(100,4510,100):
    #     print(f"Current Feature Count: {ft_cnt}")
    #     X_train = convert_data_to_tfidf(x_document,ft_cnt)
    #     y_train = numpy.array(y_label).reshape(-1, 1)
    #
    #     # first inner test by splitting the data
    #     tmp_MSE, tmp_weight = inner_split_train_test(X_train, y_train, ft_cnt)
    #     train_pred = predict(X_train, tmp_weight)
    #     tmp_acc = accuracy(train_pred, y_train)
    #     if tmp_acc > best_accuracy:
    #         best_mse_final = tmp_MSE
    #         best_ft_cnt = ft_cnt
    #         best_accuracy = tmp_acc
    #         write_weight(tmp_weight, "./weight_history", "best_weight", best_mse_final, best_ft_cnt)
    #
    # print("Best MSE: ", best_mse_final)
    # print("Best Feature Count: ", best_ft_cnt)
    # print("Best Accuracy: ", best_accuracy)

    # 0.9045833333333333
    ft_cnt = 4200
    vectorizer = TfidfVectorizer(max_features=ft_cnt)
    X_train = convert_data_to_tfidf(x_document, vectorizer)
    y_train = numpy.array(y_label).reshape(-1, 1)

    # first inner test by splitting the data
    best_mse, best_weight = inner_split_train_test(X_train, y_train, ft_cnt)
    train_pred = predict(X_train, best_weight)
    print("Training Accuracy:", accuracy(train_pred, y_train))
    # write_weight(best_weight, "./weight_history", "ultra_best_weight", best_mse, ft_cnt)

    # Final: output the test result
    # best_weight = np.array(import_weight_csv("./weight_history/ultra_best_weight_0.4771_4200.csv")).astype(float)
    file_path_for_test = "./predict_test/yprob_test.txt"
    output_test_file(x_test_doc, best_weight, file_path_for_test, vectorizer)
