import re

from tqdm import tqdm

from layers.layer import Layer


class KnowledgeOnlineGroupLayer(Layer):

    def __init__(self, debug=False):
        self.debug = debug
        pass

    def run(self, log):
        if self.debug:
            print("Prior knowledge preprocess.\n")
        value = log['Content']
        value = re.sub(
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', ' <*> ',
            value)

        # Mar 26 00:50:09
        value = re.sub(
            r'\w+ \d{2} \d{2}:\d{2}:\d{2}', ' <*> ',
            value)

        # 00:50:09.216340
        value = re.sub(
            r'\d{2}:\d{2}:\d{2}.\d{6}', ' <*> ',
            value)

        # 222.200.180.181:41406
        value = re.sub(
            r'\d+\.\d+\.\d+\.\d+:\d+', ' <*> ',
            value)

        # value = re.sub(
        #     r'\d+\.\d+\.\d+\.\d+', ' <*> ',
        #     value)

        # 英文的周数
        value = re.sub(
            r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', ' <*> ',
            value)
        # 英文的月份
        value = re.sub(
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', ' <*> ',
            value)
        log['Content'] = value

        if self.debug:
            print('Knowledge group layer finished.')
        return log
