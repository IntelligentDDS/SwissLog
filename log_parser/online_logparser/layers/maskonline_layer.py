from difflib import SequenceMatcher
from lib2to3.pgen2 import token
# from turtle import update

from tqdm import tqdm

from layers.layer import Layer
import os


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
            elif '<*>' in now.childs:
                now = now.childs['<*>']
            else:
                return -1
        return now.tag

    def update(self, old_template, template, cluster2Template, tagMap):
        if len(template) != len(old_template) or len(template) == 0:
            print("Template is Not Matching")
            return -1
        now = self.root
        
        # print(old_template, template)
        update_pos = []
        # find update position
        for i in range(len(old_template)):
            old_token = old_template[i]
            new_token = template[i]
            now = now.childs[old_token]
            if old_token != new_token and new_token == '<*>':
                now.value = '<*>'
                update_pos.append(i)

        # fing all similar templates to merge
        all_tag = self.find_cluster(template, 0, update_pos, self.root)

        for tag in all_tag:
            tagMap[tag] = now.tag

        for pos in update_pos:
            for tag in all_tag:
                if tag != -1:
                    cluster2Template[tag][pos] = '<*>'
        
        return now.tag


    def find_cluster(self, template, pos, update_pos, last):
        if pos >= len(template):
            return [last.tag]
        std_token = template[pos]
        # token = now.value
        result = list()
        if pos in update_pos:
            for token, node in last.childs.items():
                tag = self.find_cluster(template, pos + 1, update_pos, node)
                if tag:
                    result.extend(tag)
        elif std_token not in last.childs:
            return None
        else:
            result = self.find_cluster(template, pos + 1, update_pos, last.childs[std_token])

        return result         
        

    def find_cluster_id(self, childs):
        clusterID = []
        for child in childs:
            if child.tag == -1:
                clusterID.extend(self.find_cluster_id(child.childs.values()))
            else:
                clusterID.append(child.tag)
        return clusterID

    def print(self):
        now = self.root
        to_read = list()
        to_read.append(now)
        to_print = []
        while len(to_read):
            now = to_read[0]
            if now == 1:
                print(to_print)
                to_print = []
                del to_read[0]
                continue
            to_print.append(now.value)
            if now.childs != None:
                to_read.append(int(1))
                for key in now.childs.keys():
                    to_read.append(now.childs[key])
            del to_read[0]


def maskdel(template):
    temp = []
    for token in template:
        if token == '<*>':
            temp.append('')
        else:
            temp.append(token)
    return temp


class MaskOnlineLayer(Layer):
    # TODO: to split templates and cluster2Templates
    def __init__(self, dict_layer, templates: dict, results: dict, debug=False):
        self.templates = templates
        self.debug = debug
        self.orderedTemplates = dict()
        self.dict_layer = dict_layer
        self.results = results
        self.cluster2Template = dict()
        self.template2Cluster = dict()
        self.tagMap = dict()
        self.zeroTemplate = '0'
        tot = 0
        # Loading Offline Trie
        self.trie = Trie()
        for key, entry in tqdm(templates.items(), desc='constructing trie'):
            self.trie.insert(entry, tot)
            tot += 1

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
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
                result.append(seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        result = result[::-1]
        return result

    def run(self, line, log_entry):
        content = log_entry['Content']
        wordset = tuple(line['dwords'])
        if len(wordset) == 0:
            self.results[log_entry['LineId']] = 0
            self.cluster2Template[0] = self.zeroTemplate
            self.template2Cluster[self.zeroTemplate] = 0 
            return
        if wordset in self.templates:
            existTemplate = self.orderedTemplates.get(tuple(line['dwords']))
            if existTemplate is not None:
                self.results[log_entry['LineId']] = existTemplate
                return

            old_tag = self.templates[wordset]
            old_template = self.cluster2Template[old_tag]
            updated_template = self.getTemplate(
                self.LCS(content, old_template), old_template)
                
            if updated_template == old_template:
                tag = self.trie.find(updated_template)
                self.results[log_entry['LineId']] = self.tagMap[tag]
            else:
                tag = self.trie.update(old_template, updated_template,
                                    self.cluster2Template, self.tagMap)
                self.results[log_entry['LineId']] = self.tagMap[tag]
                self.trie.insert(updated_template, tag)
                self.templates[wordset] = self.tagMap[tag]
                
                # check if simpliest
                updated_templates_dict = self.dict_layer.checkValid(updated_template)
                if updated_templates_dict:
                    self.orderedTemplates[tuple(updated_templates_dict)] = self.tagMap[tag]
        else:
            tag = self.trie.find(content)
            if tag == -1:

                tag = len(self.templates) + 1
                self.trie.insert(content, tag)

                # init tag merge mapping
                self.tagMap[tag] = tag
                self.templates[wordset] = tag
                self.cluster2Template[tag] = content
                
                # check if simpliest
                updated_templates_dict = self.dict_layer.checkValid(content)
                if updated_templates_dict:
                    self.orderedTemplates[tuple(updated_templates_dict)] = tag
            else:
                self.templates[wordset] = self.tagMap[tag]
            self.results[log_entry['LineId']] = self.tagMap[tag]
