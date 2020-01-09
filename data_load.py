import codecs
import numpy as np

def load_vocab():
    vocab = ['<eos>']  #<sos>句子开始符号，<eos>句子结束符号
    file = codecs.open('data/vocab', 'r', 'utf-8')
    for line in file:
        word = line.split()[0]
        vocab.append(word)
    word2idx = {word: idx for idx, word in enumerate(vocab)}  # 词和词频：索引
    #idx2word = {idx: word for idx, word in enumerate(vocab)}  # 索引：词和词频
    fileout = codecs.open('data/word2idx', 'w', 'utf-8')
    for key in word2idx:
        fileout.write(key + '\t'+ str(word2idx[key])+'\n')
    return word2idx


def load_train():
    word2idx = load_vocab()
    x_list=[]
    file = codecs.open('data/ptb.train.txt', 'r', 'utf-8')
    for line in file:
        wordlist = (line.strip() + ' '+ '<eos>').split()
        for word in wordlist:
            x = word2idx.get(word, 2)#2是<UNK>索引
            x_list.append(x)
    return x_list


def load_test():
    word2idx = load_vocab()
    x_list=[]
    file = codecs.open('data/ptb.test.txt', 'r', 'utf-8')
    for line in file:
        wordlist = (line.strip() + ' '+ '<eos>').split()
        for word in wordlist:
            x = word2idx.get(word, 2)#2是<UNK>索引
            x_list.append(x)
    return x_list

def load_valid():
    word2idx = load_vocab()
    x_list=[]
    file = codecs.open('data/ptb.valid.txt', 'r', 'utf-8')
    for line in file:
        wordlist = (line.strip() + ' '+ '<eos>').split()
        for word in wordlist:
            x = word2idx.get(word, 2)#2是<UNK>索引
            x_list.append(x)
    return x_list

def data_iterator(raw_data, batch_size, num_steps):
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = []
    for i in range(batch_size):
        x = raw_data[batch_len * i:batch_len * (i + 1)]
        data.append(x)

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        xs = list()
        ys = list()
        for j in range(batch_size):
            x = data[j][i * num_steps:(i + 1) * num_steps]
            y = data[j][i * num_steps + 1:(i + 1) * num_steps + 1]
            xs.append(x)
            ys.append(y)
        yield (xs, ys)

