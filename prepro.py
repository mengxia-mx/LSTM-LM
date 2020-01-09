import codecs
from collections import Counter

#生成词汇文件
def statistic(inname):
    fread = codecs.open(inname,'r','utf-8').read().lstrip().rstrip()
    fread = fread.replace("\r\n", '')
    fread = fread.replace('  ', ' ')
    words = fread.split()
    word2cnt = Counter(words)
    return word2cntP

#生成词汇表
def make_vocab(outname):
    word2cnt = statistic('data/ptb.train.txt')
    with codecs.open(outname, 'w', 'utf-8') as fout:
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(word+'\t'+str(cnt)+'\n')

if __name__ == '__main__':
    make_vocab('data/vocab')
    print("生成词汇文件完成")






