from __future__ import division
from collections import defaultdict
from math import log


def train_from_file(input_file, dict_words):
    input_file = open(input_file, 'r', encoding="utf8")
    classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)
    counter = 0
    for line in input_file:
        counter += 1
        parts = line.split('\t')
        label = parts[0]
        classes[label] += 1
        for feat in get_feats(dict_words, parts[0], parts[1]):
            freq[label, feat] += 1
    for label, feat in freq:
        freq[label, feat] /= classes[label]
    for c in classes:
        classes[c] /= counter
    input_file.close()
    return classes, freq


def get_feats(dict_words, header, article):
    feats = []
    for word in header.split(' '):
        if word in dict_words:
            feats.append(word)
    for word in article.split(' '):
        if word in dict_words:
            feats.append(word)
    return feats


def classify_from_file(classify_file, result_file, classifier, dict_words):
    input_file = open(classify_file, 'r', encoding="utf8")
    output_file = open(result_file, 'w', encoding="utf8")
    classes, prob = classifier
    for line in input_file:
        parts = line.split('\t')
        feats = get_feats(dict_words, parts[0], parts[1])
        argmin = min(classes.keys(), key=lambda cl: -log(classes[cl]) + sum(-log(prob.get((cl, feat), 10 ** (-7)))
                                                                            for feat in feats))
        output_file.write(argmin + "\n")
    input_file.close()
    output_file.close()


def get_dict(dict_file):
    result = []
    input_file = open(dict_file, 'r', encoding="utf8")
    for line in input_file:
        result.append(line)
    return result


def main():
    train_file = "news_train.txt"
    dict_file = "dict.txt"
    classify_file = "news_test.txt"
    output_file = "news_output.txt"
    dict_words = get_dict(dict_file)
    classifier = train_from_file(train_file, dict_words)
    classify_from_file(classify_file, output_file, classifier, dict_words)


if __name__ == '__main__':
    main()