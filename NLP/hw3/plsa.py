import numpy as np
import math
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords


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

        rows = []
        for doc in self.documents:
            word_count = Counter(doc)
            row = []
            for word in self.vocabulary:
                row.append(word_count[word])
            rows.append(row)
        
        self.term_doc_matrix = np.array(rows)
        print("Term doc matrix:", self.term_doc_matrix)
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

        self.topic_word_prob = np.random.rand(number_of_topics, self.vocabulary_size)
        self.topic_word_prob = normalize(self.topic_word_prob)
        print("self.topic_word_prob:", self.topic_word_prob)
        print("shape:", self.topic_word_prob.shape)
        
        self.document_topic_prob = np.random.rand(self.number_of_documents, number_of_topics)
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
                prob = self.document_topic_prob[document_idx, :] * self.topic_word_prob[:, word_idx]
                prob = prob / prob.sum()
                self.topic_prob[document_idx, :, word_idx] = prob

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z) and P(z | d)
        """
        print("M step:")

        
        # update P(w | z). T, self.topic_word_prob

        # multiply term doc matrix  * topic prob (topic prob is what you calculated in E step). then you normalize over topics and words respectively.
        # when you use einsum, remember you are just updating the topic word prob and document topic prob. they should have the same dimensions as before


        # update P(z | d). D, self.document_topic_prob

        # number of times we expect to see a word in document d assigned to topic_idx j
        # n_dj = 1
        
        # Please complete this function.

        #############################
        # your code here            #
        #############################



    def calculate_likelihood(self):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        log_prob = np.log(np.matmul(self.document_topic_prob, self.topic_word_prob))
        temp = log_prob * self.term_doc_matrix
        sum = np.sum(temp)
        print("newly calculated likelihood:", sum)
        self.likelihoods.append(sum)
        return sum

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0
        
        for iteration in tqdm(range(max_iter)):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step(number_of_topics)

            previous_likelihood = current_likelihood
            current_likelihood = self.calculate_likelihood()
            current_likelihood = self.likelihoods[-1]

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
    max_iterations = 500
    # Define the convergence condition
    epsilon = 1
    corpus.plsa(number_of_topics, max_iterations, epsilon)



# print each likelihood function each iteration
if __name__ == '__main__':
    main()


# [self.vocabulary[x] for x in np.argsort(-self.topic_word_prob[0,:])[:10]]
# ['the', 'of', 'and', 'to', 'in', 'we', 'for', 'is', 'that', 'on']
# [self.vocabulary[x] for x in np.argsort(-self.topic_word_prob[1,:])[:10]]
# ['the', 'of', 'to', 'and', 'in', 'is', 'that', 'we', 'for', 'this']
