import unicodedata

import numpy as np
import torch
import re
import os
import pandas as pd
from collections import OrderedDict
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import  KFold
from sklearn.model_selection import  StratifiedKFold
import spacy
from spacy.matcher import PhraseMatcher
import pytorch_warmup as warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import requests
import nltk
from torch.utils.data import DataLoader, TensorDataset

# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('stopwords')

# init SymSpell
symSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# "frequency_dictionary_en_82_765.txt" too bad => not using anymore
#term_index=0: Specifies that the first column (index 0) in the file contains the terms (words).
#count_index=1: Specifies that the second column (index 1) in the file contains the frequency counts of the terms.
symSpell.load_dictionary("./glove_symspell_dictionary.txt", term_index=0, count_index=1)

# glove path
glove_file_path = "./glove.6B.50d.txt"
# glove as dictionary path
glove_symspell_file_path = "./glove_symspell_dictionary.txt"

# transform glove to dictionary for misspelling correction
def transform_glove_to_dictionary():
    if not os.path.exists(glove_symspell_file_path) or os.path.getsize(glove_symspell_file_path) == 0:
        with open(glove_file_path, "r") as input_file, open(glove_symspell_file_path, "w") as output_file:
            for each_line in input_file:
                first_word = each_line.split()[0]
                output_file.write(f"{first_word} 1\n")
        print("Successfully transformed glove to dictionary ✅")
    else:
        print("glove_symspell_file_path exists => skip")

# word embeddings dimension 50
def transform(sentence, word2vec, word2tfidf, dim=50):
    # pca = PCA(n_components=38)
    all_doc_tensors = []
    sentence = pd.DataFrame(sentence)
    line_series = sentence['text']
    default = np.mean(list(word2vec.values()), axis=0)
    for each_line in line_series:
        # print("Each line = ", each_line)
        each_line_words = str(each_line).split()
        each_word_embedded_vector = []
        each_word_weight = []
        for word in each_line_words:
            if word in word2vec:
                weight = word2tfidf.get(word, 5)*10
                each_word_embedded_vector.append(word2vec[word] * weight)
                each_word_weight.append(weight)
            elif "_" in word or "-" in word:
                parts = word.split("_") if "_" in word else word.split("-")
                part_vec = [word2vec[p] for p in parts if p in word2vec]
                biweight = np.mean([word2tfidf.get(p, 5)*10 for p in parts])
                if part_vec:
                    avg_vec = np.mean(part_vec, axis=0)
                    each_word_embedded_vector.append(avg_vec* biweight)
                    each_word_weight.append(biweight)
                else:
                    # not in the word2vec => set seed and use random
                    # np.random.seed(321)
                    each_word_embedded_vector.append(default * biweight)
                    each_word_weight.append(biweight)
            else:
                # use a random vector for an unknown word
                weight = word2tfidf.get(word, 0.5)
                # np.random.seed(123)
                each_word_embedded_vector.append(default * weight)
                each_word_weight.append(weight)

        # print("Each word's embedded vector = ", each_word_embedded_vector)
        if len(each_word_embedded_vector) == 0:
            each_sentence_tensor = torch.zeros(dim)
        else:
            tmp_word_arr = np.array(each_word_embedded_vector)
            # each_sentence_tensor = torch.tensor(tmp_word_arr).mean(dim=0)
            each_sentence_tensor = torch.tensor(tmp_word_arr).sum(dim=0)/sum(each_word_weight) # this is the semantic meaning of a sentence
            # each_sentence_tensor = torch.tensor(tmp_word_arr).max(dim=0).values

        # print("Each sentence tensor = ", each_sentence_tensor)
        all_doc_tensors.append(each_sentence_tensor)

    sentence_tensor = torch.stack(all_doc_tensors)
    # reduced_sentence = pca.fit_transform(sentence_tensor.numpy())
    # reduced_sentence_tensor = torch.tensor(reduced_sentence, dtype=torch.float32)
    return sentence_tensor

def max_mean_transform(sentences, word2vec, word2tfidf, dim=50):
    means, maxs = [], []
    default = np.mean(list(word2vec.values()), axis=0)
    for line in sentences:
        line_vecs, wts = [], []
        for word in str(line).split():
            if word in word2vec:
                vec  = word2vec.get(word, default)
                weight = word2tfidf.get(word, 1.0)
                line_vecs.append(vec * weight)
                wts.append(weight)
            elif "_" in word or "-" in word:
                parts = word.split("_") if "_" in word else word.split("-")
                part_vec = [word2vec.get(word, default) for p in parts if p in word2vec]
                biweight = np.mean([word2tfidf.get(p, 1.0) for p in parts])
                if part_vec:
                    avg_vec = np.mean(part_vec, axis=0)
                    line_vecs.append(avg_vec* biweight)
                    wts.append(biweight)
                else:
                    # not in the word2vec => set seed and use random
                    # np.random.seed(321)
                    line_vecs.append(default * biweight)
                    wts.append(biweight)
            else:
                # use a random vector for an unknown word
                weight = word2tfidf.get(word, 1.0)
                # np.random.seed(123)
                line_vecs.append(default * weight)
                wts.append(weight)
        if not line_vecs:
            line_vecs, wts = [default], [1.0]
        mat  = np.vstack(line_vecs)
        means.append(mat.mean(0))
        maxs.append(mat.max(0))
    sent = np.hstack([means, maxs])                       # (N, 100)
    return torch.tensor(sent, dtype=torch.float32)


# load csv file
class SentimentDataSet:
    def __init__(self, x_train_file_path, y_train_file_path,x_test_file_path,
                text_column='text', label_column='is_positive_sentiment', transform=None):
        self.word_doc_freq = None
        self.x_train_file_path = x_train_file_path
        self.y_train_file_path = y_train_file_path
        self.x_test_file_path = x_test_file_path
        self.x_col = text_column
        self.y_col = label_column
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tweet_tokenizer = TweetTokenizer()
        # custom stop words
        new_stopwords = ["oh", "yeah", "so", "just", "and"]
        self.stop_words = set(stopwords.words('english'))
        # stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
        # self.stop_words = set(stopwords_list.decode().splitlines())

        self.transform = transform
        self.max_mean_transform = max_mean_transform
        self.negation_words ={
            "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
            "can't", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't",
            "mightn't", "shan't", "n't", 'not', 'no', 'never', 'none'
        }
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self.phrases = [
            # Negation
            "not bad",
            "not good",
            "not impressed",
            "not recommend",
            "no problem",
            "no issues",
            "never again",
            "never disappointed",
            "no complaints",
            "not worth",
            "not happy",
            "not satisfied",
            "no doubt",

            # Positive
            "top notch",
            "well done",
            "highly recommend",
            "five stars",
            "exceeded expectations",
            "good value",
            "user friendly",
            "easy to use",
            "works perfectly",
            "worth every penny",
            "value for money",
            "love it",

            # Negative
            "down the drain",
            "waste of money",
            "fell apart",
            "poor quality",
            "poor product",
            "bad quality",
            "stopped working",
            "bad experience",
            "customer service",
            "never buy",
            "never work",
            "not worth it",
            "cheaply made",
            "broke after",
            "returned it",
            "would not recommend",
            "does not work",
            "out of stock",

            # Neutral
            "fast shipping",
            "as described",
            "packaging was good",
            "looks good",
            "arrived quickly",
        ]
        patterns = [self.nlp.make_doc(phrase) for phrase in self.phrases]
        self.matcher.add("IMPORTANT_PHRASES", patterns)
        self.tfidf_vectorizer = None
        self.word2tfidf = None

    def load_glove_as_dict(self):
        print("Loading glove as dictionary...")
        word_embeddings = pd.read_csv('./glove.6B.50d.txt',
                                        header=None, sep=' ', index_col=0, encoding='utf-8', quoting=3)
        # Build a dict that will map from string word to 50-dim vector
        word_list = word_embeddings.index.values.tolist()
        word2vec = OrderedDict(zip(word_list, word_embeddings.values))
        print("Type of word2vec = ", type(word2vec))
        print("len(word2vec) = ", len(word2vec))
        return word2vec

    def build_document_frequency(self, sentences):
        doc_freq = Counter()
        for sentence in sentences:
            unique_tokens = set(sentence.split())  # only count 1 per doc
            doc_freq.update(unique_tokens)
        # print(doc_freq)
        return doc_freq

    def load_train_csv(self):
        print("Loading train csv file...")
        x_doc = pd.read_csv(self.x_train_file_path)
        y_label = pd.read_csv(self.y_train_file_path)
        x_doc = x_doc[self.x_col]
        y_label = y_label[self.y_col]
        print("Cleaning text...")
        corrected_tokens = x_doc.apply(self.clean_text)
        # turn the token list into a string each row in corrected_tokens
        corrected_str = corrected_tokens.apply(lambda x: " ".join(x))
        self.word_doc_freq = self.build_document_frequency(corrected_str)
        x_doc_clean = corrected_tokens.apply(self.clean_text_with_word_doc_freq) # Series, shape:(2400,)
        print("Finished cleaning text")
        # print(f"x_doc_clean type: {type(x_doc_clean)}, shape: {x_doc_clean.shape}")
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(x_doc_clean.tolist())
        idf_values = dict(zip(self.tfidf_vectorizer.get_feature_names_out(), self.tfidf_vectorizer.idf_))
        self.word2tfidf = idf_values

        X_train = self.transform(x_doc_clean, self.load_glove_as_dict(), self.word2tfidf)
        # X_train = self.max_mean_transform(x_doc_clean, self.load_glove_as_dict(), self.word2tfidf)
        y_train = torch.tensor(y_label.values, dtype=torch.int64)
        # print("X_train shape = ", X_train.shape)
        # print("X_train = ", X_train)
        # print("y_train shape = ", y_train.shape)
        # print("y_train = ", y_train)
        return X_train, y_train, x_doc_clean

    def correct_mispronounciation(self, word):
        possible_answer = symSpell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if possible_answer:
            return possible_answer[0].term
        return word

    def tag_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def contraction_filter(self, tokens):
        filtered_tokens = []
        for token in tokens:
            if token in self.negation_words:
                filtered_tokens.append(token)
            else:
                filtered_tokens.append(token)
        return filtered_tokens

    # e.g., "down the drain" => "down_the_drain"
    def preserve_phrases(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        preserved_text = text
        for match_id, start, end in matches:
            span = doc[start:end]
            # Replace with "down_the_drain"
            preserved_text = preserved_text.replace(span.text, "_".join(span.text.split()))
        return preserved_text

    def simplify_repeats(self,text):
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def strip_accents(self, text):
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                    if unicodedata.category(c) != 'Mn')

    def clean_text(self, text):
        # print(f"Original text: {text}")
        text = text.strip()
        text = text.lower()
        text = re.sub(r"@\w+", '', text)  # remove @
        text = re.sub(r"#\w+", '', text)  # remove #
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)  # remove special characters
        text = re.sub(r'\s+', " ", text).strip()  # remove extra spaces
        text = self.simplify_repeats(text)
        text = self.strip_accents(text)
        text = self.preserve_phrases(text)
        tokens = self.tweet_tokenizer.tokenize(text)
        # tokens = word_tokenize(text)
        # tokens = self.contraction_filter(tokens)
        # correct mispronounciation
        corrected_tokens = [self.correct_mispronounciation(token) for token in tokens] # this will change "don't" to "dont"
        # print(f"Corrected tokens: {corrected_tokens}")
        # print("="*25)
        return corrected_tokens

    def clean_text_with_word_doc_freq(self, corrected_tokens):
        # print(f"Corrected tokens: {corrected_tokens}")
        # remove stop words
        no_stop_words_tokens = [token for token in corrected_tokens
                                if token not in self.stop_words or token in self.negation_words
                                and (self.word_doc_freq.get(token,0) >= 10)]
        # print(f"No stop words tokens: {no_stop_words_tokens}")
        # POS tagging
        # tagged_tokens = pos_tag(corrected_tokens)
        # word Lemmatization
        # after_lemma = [
        #             self.lemmatizer.lemmatize(each_word, self.tag_wordnet_pos(each_tag))
        #             for each_word, each_tag in tagged_tokens
        # ]
        cleaner_text = " ".join(no_stop_words_tokens)
        # print(f"Final text: {cleaner_text}")
        # print("="*25)
        return cleaner_text

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CustomNeuralNetwork,self).__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.35)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 24)
        self.bn4 = nn.BatchNorm1d(24)
        self.fc5 = nn.Linear(24, 1)
        # self.bn5 = nn.BatchNorm1d(16)
        # self.fc6 = nn.Linear(16, 1)
        # self._init_weights()
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout1(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout1(x)
        x = self.gelu(self.fc3(x))
        x = self.dropout1(x)
        x = self.gelu(self.fc4(x))
        # x = self.dropout1(x)
        # x = self.gelu(self.bn5(self.fc5(x)))
        # x = self.dropout1(x)
        y_pred = torch.sigmoid(self.fc5(x))
        return y_pred

class CustomCNN(nn.Module):
    def __init__(self, input_dim):
        super(CustomCNN, self).__init__()
        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.5)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, padding=2)  # input_channels=1, output_channels=64
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * input_dim, 1)  # Fully connected for final prediction

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # make it (batch_size, channels=1, input_dim)
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = self.gelu(self.bn3(self.conv3(x)))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # flatten for fully connected layer
        y_pred = torch.sigmoid(self.fc(x))
        return y_pred

def train_and_eval(X_train, y_train, x_doc_clean, epochs=50, eta=0.001, batch_size=32, k_folds=5, threshold=0.5):
    print("Start training...")
    seed=666
    # kf = KFold(n_splits=k_folds, shuffle=True,random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    each_fold_acc = []

    # use skf to split the data
    best_acc = 0
    no_improve = 0
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"===================== Fold {i+1} =====================")

        # get the real data from index
        X_train_fold, y_train_fold = X_train[train_index].to(torch.float32), y_train[train_index].to(torch.float32)
        X_val_fold, y_val_fold = X_train[val_index].to(torch.float32), y_train[val_index].to(torch.float32)

        # model init
        # lg_model = LogisticRegressionModel(X_train_fold.shape[1]).to(mps_device).float()
        lg_model = CustomNeuralNetwork(X_train_fold.shape[1]).to(mps_device).float()
        # lg_model = CustomCNN(X_train_fold.shape[1]).to(mps_device).float()
        bce_loss = nn.BCELoss()
        optimizer = optim.AdamW(lg_model.parameters(), lr=eta)

        scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        # Warm-up scheduler
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        for epoch in range(epochs):
            print(f"################## Epoch: {epoch+1} ##################")
            lg_model.train()
            train_dataset = TensorDataset(X_train_fold, y_train_fold.unsqueeze(1))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(mps_device), y_batch.to(mps_device)
                optimizer.zero_grad()
                y_pred = lg_model(X_batch)
                loss = bce_loss(y_pred, y_batch)
                # current batch's loss
                loss.backward()
                print(f"Fold: {i+1} Epoch:{epoch+1} Batch: {batch_idx+1} - Loss: {loss.item():.5f}")
                torch.nn.utils.clip_grad_norm_(lg_model.parameters(), 0.8)
                optimizer.step()
                # warmup_scheduler.dampen()
                scheduler_cosine.step(epoch + batch_idx / len(train_loader))

        # validation
        lg_model.eval()
        with torch.no_grad():
            y_val_pred_raw = lg_model(X_val_fold.to(mps_device)).squeeze()
            y_val_pred = (y_val_pred_raw > threshold).float()
            accuracy = (y_val_pred.cpu() == y_val_fold).to(torch.float32).mean().item()
            print(f"Fold {i+1} - Validation accuracy: {accuracy:.5f}")
            each_fold_acc.append(accuracy)
            if accuracy > best_acc:
                best_acc = accuracy
                # save the model
                torch.save(lg_model.state_dict(), f"./best_model/best_model_fold_{i+1}_{best_acc:.3f}.pth")
                print(f"Best model saved for fold {i+1} with accuracy: {best_acc:.5f}")
            else:
                no_improve += 1
                if no_improve >= 3:
                    print(f"No improvement for 5 epochs, early stopping for fold {i+1}")
                    break


    # avg accuracy of 5 folds
    avg_acc_5_fold = np.mean(each_fold_acc)
    print(f"Average accuracy of 5 folds: {avg_acc_5_fold:.5f}")
    with open("base_line_acc.txt", "a") as f:
        # Append function parameters to the accuracy file
        isCustom = True
        f.write(f"epoch: {epochs}, eta: {eta}, batch_size: {batch_size}, k_folds: {k_folds}, threshold: {threshold}, Custom: {isCustom}\n")
        f.write(f"Average accuracy of 5 folds: {avg_acc_5_fold:.5f}\n")
    return each_fold_acc

if __name__ == '__main__':
    x_train_path = "../train_data/x_train.csv"
    y_train_path = "../train_data/y_train.csv"
    x_test_path = "../test_data/x_test.csv"

    transform_glove_to_dictionary()
    X_train, y_train, x_doc_clean = SentimentDataSet(x_train_path, y_train_path, x_test_path, transform=transform).load_train_csv()
    X_train = X_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    fold5_acc = train_and_eval(X_train, y_train, x_doc_clean)
