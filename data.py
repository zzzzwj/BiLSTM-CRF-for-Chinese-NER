import pickle, random


def read_file(file_path):
    data = []
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    sent_list, tag_list = [], []
    for line in lines:
        if line != '\n':
            [char, label] = str.split(line.rstrip(), ' ')
            sent_list.append(char)
            tag_list.append(label)
        else:
            data.append((sent_list, tag_list))
            sent_list, tag_list = [], []

    return data


def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        sentence_id.append(word2id[word])
    return sentence_id


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def generate_dictionary(file_paths):
    lines = []
    data = []
    print("Reading data...")
    for file_path in file_paths:
        f = open(file_path, 'rt')
        lines += f.readlines()
    sentence = []
    tags = []
    for i, line in enumerate(lines):
        print("\r{}/{}".format(i + 1, len(lines)), end='')
        if line == '\n':
            data.append((sentence.copy(),tags.copy()))
            sentence.clear(); tags.clear()
            continue
        word, tag = line.rstrip().split(' ')
        sentence.append(word); tags.append(tag)
    print("\nDone.")

    print("Create dictionary...")
    word_to_ix = {}
    tag_to_ix = {}
    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_ix.keys():
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix.keys():
                tag_to_ix[tag] = len(tag_to_ix)

    print("Done.")

    print("Write word dictionary to file.")
    output_word = open("data/word2id.pkl", "wb")
    output_tag = open("data/tag2id.pkl", "wb")
    pickle.dump(word_to_ix, output_word)
    pickle.dump(tag_to_ix, output_tag)