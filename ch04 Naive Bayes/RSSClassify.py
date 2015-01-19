# =======================================================================
# @Author: Junbo Xin
# @Date: 2015/01/18
# @Description:  RSS Classifier using naive bayes
# =======================================================================

import bayes
from numpy import *

# Input is a big string, i.e, opening a big txt file,
# Output is a word_list to split the file
def text_parse(big_string):
    import re
    list_of_tokens = re.split(r'\W*', big_string)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


# return the 30 most frequent words in the full_text
def cal_most_freq(vocab_list, full_text):
    import operator
    freq_dict ={}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    import feedparser
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(doc_list)
        class_list.append(1)

        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(doc_list)
        class_list.append(0)

    dictionary = bayes.create_vocab_list(doc_list)
    top_30_words = cal_most_freq(dictionary, full_text)
    for word in top_30_words:
        if word[0] in dictionary:
            dictionary.remove(word[0])
    training_number = range(2*min_len)
    test_number = []

    for i in range(20):
        rand_index = int(random.uniform(0, len(training_number)))
        test_number.append(rand_index)
        del(training_number[rand_index])

    train_mat = []
    train_label = []
    for doc_index in training_number:
        train_mat.append(bayes.bag_of_vector2words(dictionary, doc_list[doc_index]))
        train_label.append(class_list[doc_index])
    p0, p1, pro_spam = bayes.train_naive_bayes(array(train_mat), array(train_label))

    error_count = 0
    for doc_index in test_number:
        word_to_vector = bayes.bag_of_vector2words(dictionary, doc_list[doc_index])
        if bayes.classify_naive_bayes(array(word_to_vector), p0, p1, pro_spam) != \
            class_list[doc_index]:
            error_count += 1
    print 'the error rate is: ', float(error_count)/ len(test_number)
    return dictionary, p0, p1

