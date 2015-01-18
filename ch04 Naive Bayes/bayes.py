# ====================================================================
# @Author: Junbo Xin
# @Date: 2015/01/18
# @Description:  Naive Bayes Classifier

# ====================================================================


def load_data_set():
    word_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ace', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]   # 0 stands for normal, 1 stands for insulting
    return word_list, class_vec


def create_vocab_list(data_set):
    vocab_list = set([])   # create empty dictionary
    for document in data_set:
        vocab_list = vocab_list | set(document)   # save only the unique words
    return list(vocab_list)


# giving the dictionary(vocab_list, input_set), return the vector(filled with 0 or 1)
def word2vec(vocab_list, input_set):
    # create the vector with same length of list, filled with 0
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index[word]] = 1
        else:
            print 'the word: %s is not in the dictionary! ' % word
    return ret_vec
