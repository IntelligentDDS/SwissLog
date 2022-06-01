# -*- coding: UTF-8 -*-
from tqdm import tqdm
# import regex as re
import re
import pandas as pd
import pickle
import wordninja


# 函数定义
def get_format(logformat):
    splitters = re.split(r'(<[^<>]+>)', logformat)
    headers = []
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return regex, headers


# 类定义
# knowledge_layer
class KnowledgeOnlineGroupLayer(object):
    def __init__(self, rex=[]):
        self.rex = rex

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, ' <*> ', line)
        return line

    def run(self, log):
        log['Content'] = self.preprocess(log['Content'])
        return log


# tokenize_group_layer
class TokenizeOnlineGroupLayer(object):
    def __init__(self):
        pass

    def splitbychars(self, s, chars):
        tokens = re.split(r'([' + chars + '])', s)
        tokens = list(filter(None, [token.strip() for token in tokens]))
        for i in range(len(tokens)):
            if all(char.isdigit() for char in tokens[i]):
                tokens[i] = '<*>'
        return tokens

    def run(self, log_entry):
        log_entry['Content'] = self.splitbychars(log_entry['Content'],
                                                 ',;:"= ')
        return log_entry


# dict_group_layer
def hasDigit(inputString):
    return any(char.isdigit() for char in inputString)


class DictOnlineGroupLayer(object):
    def __init__(self, dictionary_file=None, strip_char='.:*'):
        self.strip_char = strip_char
        self.dictionary = None
        if dictionary_file:
            with open(dictionary_file, 'rb') as f:
                self.dictionary = pickle.load(f)
        self.small_dict = dict()
        self.split_words_cache = dict()

    def checkValid(self, templates):
        final_words = []
        for word in templates:
            word = word.strip('.:*')
            if word == '':
                continue
            if hasDigit(word):
                return None
            elif not all(char.isalpha() for char in word):
                continue
            elif word in self.small_dict:
                final_words.append(word)
            elif word in self.split_words_cache:
                final_words.extend(self.split_words_cache[word])
            else:
                return None
        return tuple(final_words)

    def dictionaried(self, log_entry):
        dictionary_list = list()
        for word in log_entry['Content']:
            word = word.strip(self.strip_char)
            if hasDigit(word):
                continue
            if word == '':
                continue
            if word in self.small_dict:
                dictionary_list.append(word)
            elif word in self.dictionary:
                dictionary_list.append(word)
                self.small_dict[word] = True
            elif all(char.isalpha() for char in word):
                if word in self.split_words_cache:
                    splitted_words = self.split_words_cache[word]
                else:
                    splitted_words = wordninja.split(word)
                    self.split_words_cache[word] = splitted_words
                for sword in splitted_words:
                    if len(sword) <= 2:
                        continue
                    dictionary_list.append(sword)
        result_dict = dict(message=log_entry['Content'],
                           dwords=dictionary_list,
                           LineId=log_entry['LineId'])
        return result_dict

    def run(self, log_entry):
        return self.dictionaried(log_entry)


# mask_layer
class TreeNode:
    def __init__(self, value, tag=-1):
        self.childs = dict()  # token->treenode
        self.tag = tag  # non -1 denotes the id of the cluster
        self.value = value


class Trie:
    def __init__(self):
        self.root = TreeNode(-1)

    def insert(self, template_list, cluster_id):
        now = self.root
        for token in template_list:
            if token not in now.childs:
                now.childs[token] = TreeNode(token)
            now = now.childs[token]
        now.tag = cluster_id

    def find(self, template_list):
        now = self.root
        for token in template_list:
            # print(token, now.childs.keys())
            if token in now.childs:
                now = now.childs[token]
            else:
                return -1
        return now.tag

    def update(self, old_template, template, cluster2Template):
        now = self.root
        # print(old_template, template)
        update_pos = []
        for i in range(len(old_template)):
            old_token = old_template[i]
            new_token = template[i]
            if old_token != new_token and new_token == '<*>':
                now.value = '<*>'
                update_pos.append(i)
            now = now.childs[old_token]
        for pos in update_pos:
            cluster2Template[now.tag][pos] = '<*>'
        return now.tag

    def find_cluster_id(self, childs):
        clusterID = []
        for token, child in childs.items():
            if child.tag == -1:
                clusterID.extend(self.find_cluster_id(child.childs))
            else:
                clusterID.append(child.tag)
        return clusterID


class MaskOnlineLayer():
    def __init__(self, dict_layer, templates: dict, results: dict):
        self.dict_layer = dict_layer
        self.templates = templates
        self.results = results
        self.cluster2Template = dict()
        self.template2Cluster = dict()
        self.trie = Trie()
        # Loading Offline Trie
        # tot = 0
        # for key, entry in tqdm(templates.items(), desc='constructing trie'):
        #     self.trie.insert(entry, tot)
        #    tot += 1

    def getTemplate(self, lcs, seq):
        retVal = []
        if not lcs:
            return retVal
        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append('<*>')
            if not lcs:
                break
        while i < len(seq):
            retVal.append('<*>')
            i += 1
        return retVal

    def LCS(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2) + 1)]
                   for i in range(len(seq1) + 1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j],
                                                lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1 -
                                                        1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2
                                                                     - 1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1 - 1] == seq2[lenOfSeq2 - 1]
                result.append(seq1[lenOfSeq1 - 1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        result = result[::-1]
        return result

    def run(self, line, log_entry):
        content = log_entry['Content']
        wordset = tuple(line['dwords'])
        if wordset in self.templates:
            # print(self.templates[wordset], content)
            if self.templates[wordset] == content:
                self.results[log_entry['LineId']] = self.template2Cluster[
                    tuple(content)]
                return
            old_template = self.templates[wordset]
            updated_template = self.getTemplate(
                self.LCS(content, old_template), old_template)
            # print(old_template, updated_template)
            if updated_template == old_template:
                self.results[log_entry['LineId']] = self.template2Cluster[
                    tuple(old_template)]
                return
            tag = self.trie.update(old_template, updated_template,
                                   self.cluster2Template)
            self.results[log_entry['LineId']] = tag
            self.trie.insert(updated_template, tag)
            self.templates[wordset] = updated_template
            self.template2Cluster[tuple(updated_template)] = tag
        else:
            self.templates[wordset] = content
            tag = self.trie.find(content)
            if tag == -1:
                tag = len(self.templates)
                self.trie.insert(content, tag)
                self.cluster2Template[tag] = content
                self.template2Cluster[tuple(content)] = tag
            self.results[log_entry['LineId']] = tag


if __name__ == '__main__':
    # 文件目录结构
    # work_path = r'F:\1-重要文件学习文件备份\B - 博士研究内容（中山大学）\LOG-ALAD'
    # # 日志文件
    # log_file = work_path + r'\log\spark_injected_logs.txt'
    # # 输出文件
    # output_file = work_path + r'\output'
    # # 字典文件
    # corpus = work_path + r'\EngCorpus.pkl'
    log_file = '../logs/OpenSSH/OpenSSH_2k.log'
    output_file = './'
    corpus = './EngCorpus.pkl'
    # 日志分割格式设置
    logformat = '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>'
    regex_, headers = get_format(logformat)
    print('日志分割格式为：', headers)

    regex_setting = [
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # date xxxx-xx-xx xx:xx:xx
        r'\w+ \d{2} \d{2}:\d{2}:\d{2}',  # date Mar xx xx:xx:xx
        r'\d{2}:\d{2}:\d{2}.\d{6}',  # date xx:xx:xx.xxxxxx
        r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)',  # date week
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # date month
        r'((\d+\.){3}\d+,?)+',  # IP x.x.x.x
        r'\d+\.\d+\.\d+\.\d+:\d+',  # IP:port x.x.x.x:x
        r'/.+?\s',  # path /xxx/xxx/
        r'\s\d+\s',  # single num int
        r'(\-)?\d+(\.\d{1,2})',  # single num float
        r'(\s[0-9]+)',  # single num with blank
    ]
    knowledge_layer = KnowledgeOnlineGroupLayer(rex=regex_setting)
    tokenize_layer = TokenizeOnlineGroupLayer()
    strip_char = '.:*_'
    dict_layer = DictOnlineGroupLayer(corpus, strip_char)
    # 预读缓存
    # templates = pickle.load(open(work_path + r'\templates.pkl', 'rb'))
    templates = dict()
    results = dict()
    mask_layer = MaskOnlineLayer(dict_layer, templates, results)

    new_ = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in tqdm(fin.readlines(), desc='log parsing'):
            try:
                linecount += 1
                match = regex_.search(line.strip())
                message = dict()
                for header in headers:
                    message[header] = match.group(header)
                message['LineId'] = linecount
                content = message['Content']
                log_entry = knowledge_layer.run(message)
                log_entry = tokenize_layer.run(log_entry)
                wordset = dict_layer.run(log_entry)
                mask_layer.run(wordset, log_entry)
                new_.append([message['Date'], message['Time'], message['Timestamp'], message['Task'],
                            message['App'], message['Level'], content, results[linecount]])
                if results[linecount] == -1:
                    print(1, line, log_entry['Content'])
                    fin.close()
                    break
                # res = pd.DataFrame([[message['Date'], message['Time'], message['Timestamp'], message['Task'],
                #                     message['App'], message['Level'], message['Content'], message['Task'],
                #                     results[linecount]]])
                # res.to_csv(output_file + r'\res_log.csv', mode='a', index=False, header=False)
            except Exception as e:
                print(linecount, e, line)
                # break
                pass
    fin.close()

    # 输出模版
    template_ = []
    for idx, template in mask_layer.cluster2Template.items():
        template_.append([idx, ' '.join(template)])
    template_df = pd.DataFrame(template_, columns=['Id', 'Template'])
    template_df.to_csv(output_file + r'/res_template.csv', index=False)

    # 输出日志文件
    new_log_df = pd.DataFrame(new_, columns=['Date', 'Time', 'Timestamp', 'Task', 'App', 'Level', 'Content', 'Template'])
    new_log_df.to_csv(output_file + '/res_log.csv', index=False)
