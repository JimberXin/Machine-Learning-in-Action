'''
    @Description: Test file of bayes
'''

import bayes


def test_load():
    word_list, labels = bayes.load_data_set()
    dictionary = bayes.create_vocab_list(word_list)
    for word in dictionary:
        print word


def main():
    test_load()

if __name__ == "__main__":
    main()
