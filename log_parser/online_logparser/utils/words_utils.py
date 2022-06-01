import jieba
import jieba.posseg as pseg

jieba.enable_parallel(4)


def splitWords(str_a):
    wordsa = pseg.cut(str_a)
    seta = set()
    for key in wordsa:
        seta.add(key.word)

    return seta
