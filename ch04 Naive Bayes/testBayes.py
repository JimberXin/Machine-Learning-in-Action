'''
    @Description: Test file of bayes classifier
'''

from numpy import array
import bayes
import spamClassify
import RSSClassify


def test_load():
    word_list, labels = bayes.load_data_set()
    dictionary = bayes.create_vocab_list(word_list)
    print dictionary
    # convert each document into 0-1 vector
    train_mat = []
    for doc in word_list:
        vec = bayes.word2vec(dictionary, doc)
        train_mat.append(vec)
    p0_vec, p1_vec, prob_insult = bayes.train_naive_bayes(train_mat, labels)
    print p0_vec
    print p1_vec
    print prob_insult


def test_naive_bayes():
    word_list, labels = bayes.load_data_set()
    dictionary = bayes.create_vocab_list(word_list)
    train_mat = []
    for doc in word_list:
        vec = bayes.word2vec(dictionary, doc)
        train_mat.append(vec)
    p0_vec, p1_vec, prob_insult = bayes.train_naive_bayes(train_mat, labels)

    # test example 1:  non-spam
    test1 = ['love', 'my', 'dalmation', 'garbage']
    test1_vec = array(bayes.word2vec(dictionary, test1))
    test1_res = bayes.classify_naive_bayes(test1_vec, p0_vec, p1_vec, prob_insult)
    print test1, ': classified as: ', test1_res

    # test example 2: spam
    test2 = ['dalmation']
    test2_vec = array(bayes.word2vec(dictionary, test2))
    test2_res = bayes.classify_naive_bayes(test2_vec, p0_vec, p1_vec, prob_insult)
    print test2, ': classified as: ', test2_res


def test_spam():
    res = 0.0
    counts = 20  # test numbers
    for i in range(counts):
        res += spamClassify.spam_classify()
    print res / counts


def test_rss():
    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    RSSClassify.get_top_words(ny, sf)


def main():
    # test_load()
    # test_naive_bayes()
    # test_spam()
    test_rss()


if __name__ == "__main__":
    main()
