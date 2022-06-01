# -*- coding: utf-8 -*-
from layers.fileonline_output_layer import FileOnlineOutputLayer
from layers.knowledgeonline_layer import KnowledgeOnlineGroupLayer
from layers.maskonline_layer import MaskOnlineLayer
from layers.tokenizeonline_group_layer import TokenizeOnlineGroupLayer
from layers.dictonline_group_layer import DictOnlineGroupLayer

from evaluator import evaluator
import os
import re
from datetime import datetime
from tqdm import tqdm
import argparse
import pickle
import pandas as pd

input_dir = '../logs/' # The input directory of log file
output_dir = 'LogParserResult/' # The output directory of parsing results

def load_logs(log_file, regex, headers):
    """ Function to transform log file to dataframe
    """
    log_messages = dict()
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in tqdm(fin.readlines(), desc='load data'):
            try:
                linecount += 1
                # if linecount >3000000:
                #     break
                match = regex.search(line.strip())
                message = dict()
                for header in headers:
                    message[header] = match.group(header)
                message['LineId'] = linecount
                log_messages[linecount] = message
            except Exception as e:
                pass
    return log_messages

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
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
    return headers, regex


ds_setting = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+']
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?']
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+']
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+']
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+']
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s']
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
        },

    'Andriod': {
        'log_file': 'Andriod/Andriod_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': []
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+']
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\s?sec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']
        },


    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+', r'(\d+\.){3}\d+']
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\s\d+\s']
        },
        
    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+']
        }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', default='HDFS', type=str)
    parser.add_argument('--dictionary', default='../EngCorpus.pkl', type=str)
    # parser.add_argument('--logfile', action_store=True)

    args = parser.parse_args()
    isOnline = args.online
    debug = args.debug
    dataset = args.dataset
    corpus = args.dictionary

    # setting = ds_setting[dataset]

    benchmark_result = []
    for dataset, setting in ds_setting.items():
        print('\n=== Evaluation on %s ==='%dataset) 
        print("------------------Online Parsing------------------------\n")
        # read file settings
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        outdir = os.path.join(output_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])
        filepath = os.path.join(indir, log_file)
        print('Parsing file: ' + filepath)

        # load templates
        # templates = pickle.load(open('templates.pkl', 'rb'))
        templates = dict()
        # load log format
        headers, regex = generate_logformat_regex(setting['log_format'])
        log_messages = load_logs(filepath, regex, headers)
        # log messages is a dictionary where the key is linecount, the item is {'LineId': , header: ''}
        results = dict()
        knowledge_layer = KnowledgeOnlineGroupLayer(debug)
        tokenize_layer = TokenizeOnlineGroupLayer(rex=setting['regex'], debug=debug)
        dict_layer = DictOnlineGroupLayer(corpus, debug)
        mask_layer = MaskOnlineLayer(dict_layer, templates, results, debug)
        starttime = datetime.now()
        for lineid, log_entry in log_messages.items():
            if lineid in [1000, 10000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]:
                print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))

            # preprocess
            log_entry = knowledge_layer.run(log_entry)

            # tokenize log content
            log_entry = tokenize_layer.run(log_entry)

            # look up dictionary, return a dict: {message: log_entry['Content'], dwords: wordset, LineId: }
            wordset = dict_layer.run(log_entry)

            # LCS with existing templates, merging in prefix Tree
            mask_layer.run(wordset, log_entry)

            # print('After online parsing, templates updated: {} \n\n\n'.format(templates))
        # results = results.map(mask_layer.tagMap)
        output_file = os.path.join(outdir, log_file)
        FileOnlineOutputLayer(log_messages, results, output_file, mask_layer.cluster2Template, ['LineId'] + headers).run()
        F1_measure, accuracy = evaluator.evaluate(
                            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                            parsedresult=os.path.join(outdir, log_file + '_structured.csv')
                            )
        benchmark_result.append([dataset, F1_measure, accuracy])

    print('\n=== Overall evaluation results ===')
    avg_accr = 0
    for i in range(len(benchmark_result)):
        avg_accr += benchmark_result[i][2]
    avg_accr /= len(benchmark_result)
    pd_result = pd.DataFrame(benchmark_result, columns={'dataset', 'F1_measure', 'Accuracy'})
    print(pd_result)
    print('avarage accuracy is {}'.format(avg_accr))
    pd_result.to_csv('benchmark_result.csv', index=False)


