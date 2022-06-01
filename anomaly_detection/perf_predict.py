import torch
import torch.nn as nn
import torch.optim as optim
import gensim
import pickle
import time
import numpy
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from perf_model import BiLSTM, AC_BiLSTM, Time_BiLSTM
from perf_train import SeqDataset, PadCollate
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
window_size = 10 # sequence window length
# embed_dim = 768
num_classes = 3
# input_size = embed_dim # input vector length
hidden_size = 128 # sequence embedding size
num_layers = 1
num_epochs = 30
batch_size = 32
attention_size = 16
model_dir = 'model'
time_dim = 1
bidirectional = True
# modelName = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + ';BGL;Bert.pt'

def construct_vec(t2v, inputs):
    # print('Total size: {}*{}*{}={}'.format(len(inputs), window_size, embed_dim, len(inputs)*window_size*embed_dim))
    ret = np.zeros((len(inputs), embed_dim), np.float)
    # ret = torch.empty(len(inputs), window_size, embed_dim)
    for i in range(len(inputs)):
        # if i > 8888: continue
        eid = int(inputs[i])
        for j in range(embed_dim):
            ret[i][j] = t2v[eid][j]
    return torch.tensor(ret, dtype=torch.float)

def get_dataset(FN, s2v):
    # return list
    with open('data/' + FN, 'r') as f:
        for line in tqdm(f.readlines()):
            # self.num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            vecs = []
            for i in range(len(line)):
                eid = int(line[i]) + 1
                vecs.append(s2v[eid])
    return vecs 
    # hdfs = set()
    # with open('../data/' + name, 'r') as f:
    #     for line in f.readlines():
    #         # line = list(map(lambda n: n - 1, map(int, line.strip().split())))
    #         line = list(map(int, line.strip().split()))
    #         if len(line) < 1: continue
    #         hdfs.add(tuple(line))
    # print('Number of sessions({}): {}'.format(name, len(hdfs)))
    # return hdfs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-encoding', default='hdfs_sentence2vec.pkl', type=str)
    parser.add_argument('-embeddim', default=300, type=int)
    parser.add_argument('-dataset', default='hdfs', type=str)
    parser.add_argument('-type', default='float32', type=str)
    parser.add_argument('-attn', default=True, type=bool)
    parser.add_argument('-timedim', default=300, type=int)

    
    parser.add_argument('-model', default='Adam_batch_size=32;epoch=30;BGL;tf-idf.pt', type=str)
    parser.add_argument('-indir', default='data/hdfs', type=str)
    parser.add_argument('-bi', default=True, type=bool)
    parser.add_argument('-time', default='model/time_embedding.pt', type=str)
    # parser.add_argument('-perf', default=False, type=bool)
    # parser.add_argument('-window_size', default=10, type=int)
    args = parser.parse_args()
    encodingFN = args.encoding
    embed_dim = args.embeddim
    datasetName = args.dataset
    modelName = args.model
    tensortype = args.type
    indir = args.indir
    bi_flag = args.bi
    timeembedding=args.time
    timedim = args.timedim
    # perf = args.perf
    attnFlag = args.attn
#    import pdb; pdb.set_trace()
    
    if tensortype == 'float32':
        tensortype = torch.float32
    else:
        tensortype = torch.double
    
    s2v = pickle.load(open(encodingFN, 'rb'))
    model = BiLSTM(batch_size, embed_dim, hidden_size, num_layers, num_classes,  bi_flag, True, attnFlag, timedim, device)
   # model = nn.DataParallel(model)
    model.to(device)
    #model = AC_BiLSTM(embed_dim, hidden_size, num_layers, num_classes, bi_flag, device).to(device)
    dataloader = DataLoader(SeqDataset(indir + '/my_'+ datasetName + '_test_normal', indir + '/my_'+ datasetName + '_test_abnormal', indir + '/my_' + datasetName + '_test_perf', encodingFN), batch_size=batch_size, shuffle=True, pin_memory=False, collate_fn=PadCollate(dim=0, typ=tensortype))
    model.load_state_dict(torch.load(modelName))
    model.eval()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    start_time = time.time()
    label = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, (seq, timedelta, label, leng) in tqdm(enumerate(dataloader)):
            y, output, seq_len = model(seq.to(device), timedelta.to(device), label.to(device), leng.to(device))
#            import pdb; pdb.set_trace()
            _, preds = torch.max(output, 1)
            for i in range(len(preds)):
                y_true.append(y[i])
                y_pred.append(preds[i])
    #import pdb; pdb.set_trace()

    y_true = [i.item() for i in y_true]
    y_pred = [i.item() for i in y_pred]
    f1 = f1_score( y_true, y_pred, average='macro' )
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    
    # Compute precision, recall and F1-measure
    # P = 100 * TP / (TP + FP)
    # R = 100 * TP / (TP + FN)
    # print(p)
    # print(r)
    # # F1 = 2 * P * R / (P + R)
    print('model:', modelName)
    print('dataset:', indir)
    print('classification report', classification_report(y_true, y_pred))
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(p*100, r*100, f1*100))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
