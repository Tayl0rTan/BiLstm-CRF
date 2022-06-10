# -*- encoding: utf-8 -*-

vocab = {}



def load_tag():
    tag = {}
    with open("data/label.txt") as reader:
        line = reader.readline().strip()
        while line:
            tag[line] = len(tag)+1
            line = reader.readline().strip()
    return tag


def load_vocab():
    vocab = {}
    with open("data/vocab.txt") as reader:
        line = reader.readline().strip('\n')
        while line:
            vocab[line] = len(vocab)
            line = reader.readline().strip('\n')
    return vocab


def text2Int(text, vocab):
    ints = []
    for term in text:
        try:
            x = vocab[term]
        except:
            x = 0
        ints.append(x)
    return ints


def load_data(filename):
    #[call_id, speak_channel, speak_index, text,label]
    data = []
    labels = []
    vocab = load_vocab()
    tag = load_tag()
    print(vocab)
    with open(filename, "r") as reader:
        line = reader.readline().strip()
        call = None
        while line:
            line_ = line.split("\t")
            if len(line_) != 5:
                raise Exception("Data Format Wrong")
            call_id = line_[0]
            textInts = text2Int(list(line_[3]), vocab)
            try:
                labelInt = tag[line_[4]]
            except:
                labelInt = 0
            if not call:
                text_ = [textInts]
                label_ = [labelInt]
            elif call_id == call:
                text_.append(textInts)
                label_.append(labelInt)
            else:
                data.append(text_)
                labels.append(label_)
                text_ = [textInts]
                label_ = [labelInt]
            call = call_id
            line = reader.readline().strip()
    return data, labels


if __name__ == '__main__':
    data,labels = load_data('data/ner_test.txt')
    print(data)
    print(labels)
    for i in range(len(data)):
        print(len(data[i]), len(labels[i]))
