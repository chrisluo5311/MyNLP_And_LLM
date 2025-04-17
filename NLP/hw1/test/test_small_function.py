import numpy as np
from symspellpy import SymSpell, Verbosity

def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

def correct_words():
    # init SymSpell
    symSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    symSpell.load_dictionary("../frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

    # input text
    wrong_str_list = ["Absolutel junk",
                    "This is a simple little phone to use, but the breakage is unacceptible.",
                    "This is so embarassing and also my ears hurt if I try to push the ear plug into my ear."]

    # tokenize and correct each word
    for line in wrong_str_list:
        corrected_str = []
        for word in line.split():
            suggestions = symSpell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                corrected_str.append(suggestions[0].term)
            else:
                corrected_str.append(word)
        corrected_text = ' '.join(corrected_str)
        print(corrected_text)

if __name__ == '__main__':
    # a = np.array([1,3,5])
    # b = np.array([0,-1,2])
    # print(euclidean_distance(a, b))

    yprob_test = np.loadtxt('../predict_test/yprob_test.txt')
    if np.any(yprob_test > 1) or np.any(yprob_test < 0):
        print("yprob_test is out of range❗️")
    else:
        print("yprob_test is All in range👌")
    # print(yprob_test)
    print(yprob_test.shape)
    print(type(yprob_test))

    # correct_str = str(TextBlob(wrong_str).correct())
    # print(correct_str)

    # correct_words()