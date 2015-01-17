import trees

if __name__ == '__main__':
    data_set, labels = trees.create_data_set()
    entropy = trees.calc_shannon_entropy(data_set)
    ret = trees.split_data_set(data_set, 1, 1)
    feature = trees.choose_features(data_set)
    my_tree = trees.create_tree(data_set, labels)
    print my_tree
