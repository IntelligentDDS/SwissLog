from tqdm import tqdm
import pickle
from bert_serving.client import BertClient


t2wPath = '../data/template2words.pkl'
outputPath = '../data/bert_encoding.pkl'

bc = BertClient() 
with open(t2wPath, 'rb') as f:
    data = pickle.load(f)
# group all words into one sentence
templateSentence = dict()
for i, v in tqdm(data.items()):
    sen = ' '.join(v)  
    templateSentence[i] = bc.encode(sen)

with open(outputPath, 'wb') as f:
    pickle.dump(templateSentence)

print('Successfully Finished BERT Encoding')
