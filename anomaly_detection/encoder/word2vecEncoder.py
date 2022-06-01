from tqdm import tqdm
import csv
import pickle
import re
import random
import fasttext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import gensim
import argparse

class SemanticVec:

    def __init__(self, w2vmodel, t2wFN, embed_dim):
        # self.corpus = corpus
        self.embed_dim = embed_dim
        self.w2vmodel = w2vmodel
        self.t2wFN = t2wFN
        self.stop_words = ['a', 'the']

    def csr_todense(self, n_row, n_col, Ap, Aj, Ax):
        result = np.zeros([n_row, n_col])
        for i in range(0, n_row):
            for j in range(Ap[i], Ap[i + 1]):
                result[Aj[j]] += Ax[j]
        return result

    def run(self):
        # loadtemplate()
        # t2w = gensim.models.KeyedVectors.load_word2vec_format(self.w2vmodel, binary=False, encoding='utf8')
        t2w = pickle.load(open(self.t2wFN, 'rb'))
        # preprocess()
        # import pdb; pdb.set_trace()
        self.preprocess(t2w)
        # word2vec()
        self.word2vec()
        # calTFIDF()
        weight, vectorizer = self.calTFIDF()
        senvec = dict()
        print('Semantic Vectorization...')
        #import pdb; pdb.set_trace()
        i = 0
        for idx, wvs in tqdm(self.vecs.items()):
            #print(idx)
         #   import pdb; pdb.set_trace()
            vec = np.zeros(self.embed_dim)
            w = weight[i]
            i += 1
            v = np.zeros(self.embed_dim)
            for ii, wv in enumerate(wvs):
                wi = vectorizer.vocabulary_.get(t2w[idx][ii])
                # if type(wi) is np.int64:
                if type(wi) is int:
                    ww = w[wi]
                else:
                    ww = 0
                v = v + np.array(ww * wv)
            v = v.astype('float64')
            senvec[idx] = v
            print(v.dtype)
#        print(senvec.shape)
        return senvec


    def preprocess(self, t2w):
        print('Preprocessing...')
        self.t2w_filter = dict()
        for t, words in tqdm(t2w.items()):
            print(t)
            print(words)
            self.t2w_filter[t] = [word for word in words if word not in self.stop_words]

    def word2vec(self):
        print('Word Embedding...')
        # pre-trained on Common Crawl Corpus dataset using the FastText algorithm
        model = fasttext.load_model(self.w2vmodel)
        # model = gensim.models.KeyedVectors.load_word2vec_format(self.w2vmodel, binary=False, encoding='utf8')
        self.vecs = dict()
        for idx, line in tqdm(self.t2w_filter.items()):
            vec = list()
            for word in line:
                vec.append(model[word])
            self.vecs[idx] = vec

    def calTFIDF(self):
        print('Calculating TFIDF weight...')
        vectorizer = CountVectorizer()
        # 计算词语出现的次数
        corpus = list()
        for idx, lines in self.t2w_filter.items():
            corpus.append(' '.join(lines))
            # corpus = [l for idx, lines in self.t2w_filter.iterms() l = ' '.join(lines)]
        X = vectorizer.fit_transform(corpus)
        # 获取磁带中所有文本关键词
        # word = vectorizer.get_feature_names()
        # 类调用
        # import pdb; pdb.set_trace()
        transformer = TfidfTransformer()
        # print transformer
        # 将词频矩阵X统计成TF-IDF值
        # import ipdb; ipdb.set_trace()
        tfidf = transformer.fit_transform(X)
        M, N = tfidf._swap(tfidf.shape)
        # weight = self.csr_todense(M, N, tfidf.indptr, tfidf.indices, tfidf.data)
        weight = tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        # import ipdb; ipdb.set_trace()
        return weight, vectorizer

parser = argparse.ArgumentParser()
parser.add_argument('-ratio', default=1, type=float)
args = parser.parse_args()
ratio =  args.ratio
w2vmodelPath = '../pre-trained-model/crawl-300d-2M-subword.bin'
t2wPath = '../data/container_0319/template2words.pkl'
#t2wPath = '../data/hdfs_' + str(ratio) + '/my_hdfs_' + str(ratio) + '_template2words.pkl'
embed_dim = 300
model = SemanticVec(w2vmodelPath, t2wPath, embed_dim)
vecs = model.run()
pickle.dump(vecs, open('../data/container_0319/sentence2vec.pkl', 'wb'))
#pickle.dump(vecs, open('../data/hdfs_' + str(ratio) + '/sentence2vec.pkl', 'wb'))
