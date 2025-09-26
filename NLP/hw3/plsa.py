import numpy as np
import math
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import os
import random


# nltk.download('stopwords')

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """
    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    # print("rowsums",row_sums)
    new_matrix = input_matrix / row_sums[:, np.newaxis]

    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        file = open(self.documents_path, 'r')

        lines = file.readlines()
        for line in lines:
            if line[0] in ["0","1"]: # document is labeled 0 (seattle) or 1 (chicago) for its topic
                line = line[2:]
            words = line.strip().split(" ")
            self.documents.append(words)
        
        self.number_of_documents = len(self.documents)
        file.close()

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        stop_words = set(stopwords.words('english'))
        for doc in self.documents:
            for word in doc:
                if word.lower() not in stop_words:
                    if word not in self.vocabulary:
                        self.vocabulary.append(word)
                        self.vocabulary_size += 1

        self.vocabulary_size = len(self.vocabulary)

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        matrix_file = "term_doc_matrix.npy"
        if os.path.exists(matrix_file) and os.path.getsize(matrix_file) > 0:
            self.term_doc_matrix = np.load(matrix_file)
            print("Loaded term doc matrix from file.")
            print("Term doc matrix shape:", self.term_doc_matrix.shape)
            return

        rows = []
        for doc in self.documents:
            word_count = Counter(doc)
            row = []
            for word in self.vocabulary:
                row.append(word_count[word])
            rows.append(row)
        
        self.term_doc_matrix = np.array(rows)
        np.save(matrix_file, self.term_doc_matrix)
        print("Term doc matrix:", self.term_doc_matrix) # 1000x9863
        print("Term doc matrix shape:", self.term_doc_matrix.shape)

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # T : topic_word_prob (topics by words). theta_i's. Each element is the probability of a particular word in a particular topic.
        # D : document_topic_prob (documents by topics). p_ij's. Each element is the probability that a particular topic is covered in a particular document.

        # self.document_topic_prob = None  # P(z | d)
        # self.topic_word_prob = None  # P(w | z)
        # self.topic_prob = None  # P(z | d, w)

        self.topic_word_prob = np.random.rand(number_of_topics, self.vocabulary_size) # P(w|z) shape: 8x9863
        self.topic_word_prob = normalize(self.topic_word_prob)
        print("self.topic_word_prob:", self.topic_word_prob)
        print("shape:", self.topic_word_prob.shape)
        
        self.document_topic_prob = np.random.rand(self.number_of_documents, number_of_topics) # P(z|d) shape: 1000x8
        self.document_topic_prob = normalize(self.document_topic_prob)
        print("self.document_topic_pro:", self.document_topic_prob)
        print("shape:", self.document_topic_prob.shape)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d) self.topic_prob
        """
        print("E step:")

        for word_idx in tqdm(range(self.vocabulary_size)):
            for document_idx in range(self.number_of_documents):
                # 1000x8 * 8x9863 => [1,8] * [8x1] => [8,]
                prob = self.document_topic_prob[document_idx, :] * self.topic_word_prob[:, word_idx]
                sum_prob = prob.sum()
                if sum_prob == 0:
                    prob = np.ones_like(prob) / len(prob)
                else:
                    prob = prob / sum_prob
                self.topic_prob[document_idx, :, word_idx] = prob

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z) and P(z | d)
        """
        print("M step:")

        
        # update P(w | z). T, self.topic_word_prob

        # multiply term doc matrix(1000x9863) * topic prob (topic prob is what you calculated in E step). then you normalize over topics and words respectively.
        # when you use einsum, remember you are just updating the topic word prob and document topic prob. they should have the same dimensions as before

        w_z_prob = np.einsum('dw,dzw->zw', self.term_doc_matrix, self.topic_prob) # 1000x9863, 1000x8x9863 => 8x9863
        z_prob = np.einsum('dw,dzw->z', self.term_doc_matrix, self.topic_prob) # 1000x9863, 1000x8x9863 => 8,
        #z_prob = np.where(z_prob == 0, 1e-12, z_prob) # avoid division by zero
        self.topic_word_prob = w_z_prob/z_prob[:, np.newaxis] # 8x9863 / 8 => 8x9863

        # update P(z | d). D, self.document_topic_prob

        # number of times we expect to see a word in document d assigned to topic_idx j
        # n_dj = 1

        z_d_prob = np.einsum('dw,dzw->dz', self.term_doc_matrix, self.topic_prob) # 1000x9863, 1000x8x9863 => 1000,8
        d_prob = self.term_doc_matrix.sum(axis=1, keepdims=True) # 1000x9863 => [1000,1]
        #d_prob = np.where(d_prob == 0, 1e-12, d_prob)
        self.document_topic_prob = z_d_prob/d_prob # 1000x8 / 1000 => 1000x8


    def calculate_likelihood(self):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        dot = np.matmul(self.document_topic_prob, self.topic_word_prob)
        dot = np.where(dot == 0, 1e-12, dot) # avoid division by zero
        log_prob = np.log(dot)
        temp = log_prob * self.term_doc_matrix
        total_likelihood = np.sum(temp)
        print("newly calculated likelihood:", total_likelihood)
        self.likelihoods.append(total_likelihood)
        return total_likelihood

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w) => 在文件 d 中某個位置觀察到字詞 w 時，該位置所屬的主題為 z 的後驗機率
        # 1000x8x9863
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        log_file_path = "likelihood_log.log"
        with open(log_file_path, 'w') as log_file:
            log_file.write("Iteration\tLikelihood\n")

        for iteration in tqdm(range(max_iter)):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step(number_of_topics)

            previous_likelihood = current_likelihood
            current_likelihood = self.calculate_likelihood()
            current_likelihood = self.likelihoods[-1]

            with open(log_file_path, "a") as f:
                f.write(f"{iteration + 1}\t{current_likelihood}\n")

            for i in range(number_of_topics):
                print("Top 10 words in topic {}: {}".format(i+1, [self.vocabulary[x] for x in np.argsort(-self.topic_word_prob[i,:])[:10]]))

            if abs(previous_likelihood - current_likelihood) < epsilon:
                print("yeah this is pretty good.")
                print(current_likelihood)
                return current_likelihood
            else:
                print("Current likelihood is ", current_likelihood)



def main():
    documents_path = 'DBLP_1000.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    # Define the topics numbers
    number_of_topics = 8
    # Define the iteration number
    max_iterations = 375
    # Define the convergence condition
    epsilon = 1
    corpus.plsa(number_of_topics, max_iterations, epsilon)
    return corpus

def print_top_words_from_sampled_topics(corpus, number_of_samples=2, top_k=10, output_file="sampled_topics.txt"):
    random.seed(666)
    print("\nSampling and printing top words for topics...\n")
    sampled_topic_indices = random.sample(range(corpus.topic_word_prob.shape[0]), number_of_samples)

    with open(output_file, "w") as f:
        f.write("Sampled Topics - Top Words Report\n")
        f.write("=" * 50 + "\n\n")

        for topic_idx in sampled_topic_indices:
            top_indices = np.argsort(-corpus.topic_word_prob[topic_idx])[:top_k]
            top_words = [corpus.vocabulary[i] for i in top_indices]
            top_probs = corpus.topic_word_prob[topic_idx][top_indices]

            f.write(f"Topic {topic_idx + 1}:\n")
            for word, prob in zip(top_words, top_probs):
                f.write(f"{word:20s} {prob:.6f}\n")
            f.write("\n" + "-" * 50 + "\n\n")

    print(f"\nTop words for {number_of_samples} sampled topics saved to '{output_file}'\n")


# print each likelihood function each iteration
if __name__ == '__main__':
    corpus = main()
    print_top_words_from_sampled_topics(corpus)


# [self.vocabulary[x] for x in np.argsort(-self.topic_word_prob[0,:])[:10]]
# ['the', 'of', 'and', 'to', 'in', 'we', 'for', 'is', 'that', 'on']
# [self.vocabulary[x] for x in np.argsort(-self.topic_word_prob[1,:])[:10]]
# ['the', 'of', 'to', 'and', 'in', 'is', 'that', 'we', 'for', 'this']
