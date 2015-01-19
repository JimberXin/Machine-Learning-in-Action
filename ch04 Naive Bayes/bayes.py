# ====================================================================
# @Author: Junbo Xin
# @Date: 2015/01/18
# @Description:  Naive Bayes Classifier

# ====================================================================

from numpy import *


# create the word lists and their labels
def load_data_set():
    word_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ace', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]   # 0 stands for normal, 1 stands for insulting
    return word_list, class_vec


# create dictionary: all words are unique
def create_vocab_list(data_set):
    vocab_list = set([])   # create empty dictionary
    for document in data_set:
        vocab_list = vocab_list | set(document)   # save only the unique words
    return list(vocab_list)


# giving the dictionary and a new word list, return the word's vector(fill with 0 or 1)
def word2vec(vocab_list, input_set):
    # create the vector with same length of list, filled with 0
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] = 1
        else:
            print 'the word: %s is not in the dictionary! ' % word
    return ret_vec


# Using bag-of-word model
def bag_of_vector2words(vocab_list, input_set):
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] += 1   # only difference with set-of-word model
    return ret_vec


# Giving a 0-1 training matrix and their corresponding labels ,return each word's prob
def train_naive_bayes(train_matrix, train_label):
    # step 1: calculate the num of the doc and the total words in the dictionary
    num_of_doc = len(train_matrix)
    num_of_word = len(train_matrix[0])

    # step 2: calculate the probability of classes, here calc the insulting class
    prob_insult = sum(train_label) / float(num_of_doc)

    # step 3: initialize numerator and denominator
    p0_numerator = ones(num_of_word)    # changes to ones(), in case it's close to 0
    p1_numerator = ones(num_of_word)
    p0_denominator = 2.0                # changes 0.0 to 2.0, in case it's very small
    p1_denominator = 2.0

    # step 4: for p0_vec and p1_vec, each elements stand for it's conditional probability
    #         p0_vec = [p(w1|c0), p(w2|c0), p(w3|c0), ..., p(wn|c0)]
    for i in range(num_of_doc):
        if train_label[i] == 1:
            p1_numerator += train_matrix[i]   # add each word in the document
            p1_denominator += sum(train_matrix[i])
        else:
            p0_numerator += train_matrix[i]
            p0_denominator += sum(train_matrix[i])
    p0_vec = log(p0_numerator/p0_denominator)       # change to log()
    p1_vec = log(p1_numerator/p1_denominator)       # change to log()
    return p0_vec, p1_vec, prob_insult


# giving the test word vector: vec_to_test, classify it as insult(1) or not(0)
def classify_naive_bayes(vec_to_test, p0_vec, p1_vec, prob_insult):
    # base on the NB:  p(c_i|w) = p(w_1|c_i)p(w_2|c_i)...p(w_n|c_i) * p(c_i) / p(w)
    p1 = sum(vec_to_test * p1_vec) + log(prob_insult)
    p0 = sum(vec_to_test * p0_vec) + log(1.0 - prob_insult)
    if p1 > p0:
        return 1
    else:
        return 0





