import torch
import torch.nn as nn
import torch.optim as optim
import gensim
import pickle
import time
import numpy
from tqdm import tqdm
import os
import torch.utils.data as data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from perf_model import LogRobustModel, BiLSTM, AC_BiLSTM, Time_BiLSTM
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

os.environ['CUDA_VISIBLE_DEVICES']='2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
window_size = 10 # sequence window length
# embed_dim = 300
num_classes = 3
# input_size = embed_dim # input vector length
hidden_size = 128 # sequence embedding size
num_layers = 1
num_epochs = 30
batch_size = 32
attention_size = 16
model_dir = 'model'
time_dim = 1
#bidirectional = True
log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + 'blk'


class SeqDataset(data.Dataset):
    def __init__(self, normalFN, abnormalFN, perfFN, s2vFN):
        self.num_sessions = 0
        self.inputs = []
        self.outputs = []
        self.lengths = []
        self.timedelta = []
        # normalFN = 'my_hdfs_train_normal'
        # abnormalFN = 'my_hdfs_train_abnormal'
        #if not perf:
        self.cnt = 0
        self.time_sum = 0
        self.s2v = pickle.load(open(s2vFN, 'rb'))
        #self.s2va = pickle.load(open('data/my_dataset/LogInsight_merged_hdfs_bert_cased_ori.pkl', 'rb'))
        # seq_dataset = get_dataset('my_hdfs_train_normal', 'my_hdfs_train_abnormal', s2v)
        self.dataLabeled(normalFN, 0)
        self.dataLabeled(abnormalFN, 1)
        self.dataLabeled(perfFN, 2)
        #import pdb; pdb.set_trace()
        # record by list, following time labeled should follow the same order   
        self.time_dataLabeled(normalFN + '_time')
        self.time_dataLabeled(abnormalFN + '_time')
        self.time_dataLabeled(perfFN + '_time')
        #print('max inputs length: ', max(len(self.inputs[i]) for i in range(len(self.inputs))))
        self.standardscaler()
        # return inputs, outputs
        # mms = MinMaxScaler()
        # self.timedelta = mms.fit_transform(self.timedelta)
    def standardscaler(self):
        mean = self.time_sum / self.cnt
        se = 0
        for i in self.timedelta:
            for j in i:
                se += (j - mean) ** 2
        se = np.sqrt(se / (self.cnt - 1))
        
        for i in range(len(self.timedelta)):
            for j in range(len(self.timedelta[i])):
                scale = (self.timedelta[i][j] - mean) / se
                self.timedelta[i][j] = scale
        
    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.timedelta[index]), torch.tensor(self.outputs[index]), torch.tensor(self.lengths[index])


    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.outputs)

    def dataLabeled(self, FN, label):
        with open(FN, 'r') as f:
            for line in tqdm(f.readlines()):
                self.num_sessions += 1
                line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
                vecs = []
#                import pdb; pdb.set_trace()
                for i in range(len(line)):
                    #if self.perf:
                    #    vecs.append(line[i])
                    #else:
                    eid = int(line[i]) + 1
                    vecs.append(self.s2v[eid])
                # for i in range(len(line) - window_size):
                self.inputs.append(vecs)# [len, emd]
                self.outputs.append(label)
                self.lengths.append(len(line))

    def dataLabeleda(self, FN, label):
        with open(FN, 'r') as f:
            for line in tqdm(f.readlines()):
                self.num_sessions += 1
                line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
                vecs = []
#                import pdb; pdb.set_trace()
                for i in range(len(line)):
                    #if self.perf:
                    #    vecs.append(line[i])
                    #else:
                    eid = int(line[i]) + 1
                    vecs.append(self.s2va[eid])
                # for i in range(len(line) - window_size):
                self.inputs.append(vecs)# [len, emd]
                self.outputs.append(label)
                self.lengths.append(len(line))
    def time_dataLabeled(self, FN):
        with open(FN, 'r') as f:
            for line in tqdm(f.readlines()):
                #import pdb; pdb.set_trace()
                self.num_sessions += 1
                line = tuple(map(lambda n: n, map(int, line.strip().split())))
                #line = line.strip().split()
                td = [-1]
                for i in range(len(line)):
                    td.append(line[i])
                    # if line[i] == 0:
                    #     td.append(1.5)
                    # else:
                    #     td.append(1 / line[i])
                    self.cnt += len(line)
                self.time_sum += sum(td)
                #td = [-1] + [ line[i] for i in range(len(line))]
                self.timedelta.append(td) # [len]



def pad_tensor(vec, pad, dim, typ):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    if (vec.dtype==torch.int64):
        vec = torch.tensor(vec, dtype=torch.float32)

    return torch.cat([vec, torch.zeros(*pad_size, dtype=typ)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0, typ=torch.float32):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.type = typ

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label, length)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
            ls - a tensor of all lengths in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # print('max len', max_len)
        # pad according to max_len
#        pad_batch = map(lambda x: pad_tensor(x[0], pad=max_len, dim=self.dim), batch)
        pad_batch = []
        for i in range(len(batch)):
            pad_batch.append(pad_tensor(batch[i][0], pad=max_len, dim=self.dim, typ=self.type))
        xs = torch.stack(pad_batch, dim=0) # [b, ml, e]
        pad_batch_2 = [] 
        # import pdb; pdb.set_trace()
        for i in range(len(batch)):
            pad_batch_2.append(pad_tensor(batch[i][1], pad=max_len, dim=self.dim, typ=torch.float32))
        txs = torch.stack(pad_batch_2, dim=0) # [b, ml]
        ys = torch.tensor(list(map(lambda x: x[2], batch))) # [b, 1]
        ls = torch.tensor(list(map(lambda x: x[3], batch))) # [b, 1]
        return xs, txs, ys, ls

    def __call__(self, batch):
        return self.pad_collate(batch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-encoding', default='hdfs_sentence2vec.pkl', type=str)
    parser.add_argument('-embeddim', default=300, type=int)
    parser.add_argument('-timedim', default=300, type=int)
    parser.add_argument('-dataset', default='hdfs', type=str)
    parser.add_argument('-type', default=10, type=str)
    parser.add_argument('-output', default='model/default.pt', type=str)
    parser.add_argument('-bi', default='True', type=bool)
    parser.add_argument('-attn', default='True', type=bool)
    parser.add_argument('-indir', default='data/LogInsight/hdfs', type=str)
    args = parser.parse_args()
    encodingFN = args.encoding
    time_dim = args.timedim
    embed_dim = args.embeddim
    tensortype = args.type
    outputPath = args.output
    bidirectional = args.bi
    indir = args.indir
    attnFlag = args.attn
#    import pdb; pdb.set_trace()
    datasetName = args.dataset
    if tensortype == 'float32':
        tensortype = torch.float32
    elif tensortype == 'double':
        tensortype = torch.double
    #elif tensortype == 'long':
   #     tensortype = torch.long
    model = BiLSTM(batch_size, embed_dim, hidden_size, num_layers, num_classes,  bidirectional, True, attnFlag, time_dim, device).to(device)
    #model = AC_BiLSTM(embed_dim, hidden_size, num_layers, num_classes, bidirectional, device).to(device)
    dataloader = DataLoader(SeqDataset(indir + '/my_'+ datasetName + '_train_normal', indir + '/my_'+ datasetName + '_train_abnormal', indir + '/my_' + datasetName + '_train_perf', encodingFN), batch_size=batch_size, shuffle=True, pin_memory=False, collate_fn=PadCollate(dim=0, typ=tensortype))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
#    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    #import pdb; pdb.set_trace()
    #for step, fk in enumerate(dataloader):
   # import pdb; pdb.set_trace()
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, timedelta, label, leng) in enumerate(dataloader):
            #import pdb; pdb.set_trace()
            y, output, seq_len = model(seq.to(device), timedelta.to(device), label.to(device), leng.to(device))

            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], Train_loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))
    #import pdb; pdb.set_trace()
  #  if not os.path.isdir(model_dir):
   #     os.makedirs(model_dir)
    torch.save(model.state_dict(), outputPath)
    #torch.save(model.timeembedding, 'model/time_embedding.pt')
    print('Finished Training')
