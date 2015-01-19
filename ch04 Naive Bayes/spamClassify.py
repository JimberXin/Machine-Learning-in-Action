# =======================================================================
# @Author: Junbo Xin
# @Date: 2015/01/18
# @Description:  Spam Classifier using naive bayes

# =======================================================================

import bayes
from numpy import *


# Input is a big string, i.e, opening a big txt file,
# Output is a word_list to split the file
def text_parse(big_string):
    import re
    list_of_tokens = re.split(r'\W*', big_string)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


# Giving 25 spam emails and 25 ham emails
def spam_classify():
    # step 1: create the document list
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        # deal with spam email
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)   # spam email labeled as: 1

        # deal with ham email
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)   # ham email labeled as: 0

    # step 2: create the dictionary: unique words
    dictionary = bayes.create_vocab_list(doc_list)

    # step 3: create train set and train label
    # randomly select 10 txt for test, 40 txt for training
    training_number = range(50)
    test_number = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_number)))
        test_number.append(training_number[rand_index])
        del(training_number[rand_index])
    train_mat = []
    train_label = []
    for doc_index in training_number:
        # important! use bag of words, not set of words
        train_mat.append(bayes.bag_of_vector2words(dictionary, doc_list[doc_index]))
        train_label.append(class_list[doc_index])
    p0_vec, p1_vec, prob_spam = bayes.train_naive_bayes(array(train_mat), array(train_label))

    # step 4: use the classifier to test the email
    error_count = 0
    for doc_index in test_number:
        word_to_vector = bayes.bag_of_vector2words(dictionary, doc_list[doc_index])
        if bayes.classify_naive_bayes(array(word_to_vector), p0_vec, p1_vec, prob_spam) != \
           class_list[doc_index]:
                error_count += 1
                # print 'Here is the vector', word_to_vector
                print 'Come from: %d: %d %s' % (doc_index,  class_list[doc_index], doc_list[doc_index])
    print 'the current error rate is: ', float(error_count)/len(test_number)
    return float(error_count) / len(test_number)










