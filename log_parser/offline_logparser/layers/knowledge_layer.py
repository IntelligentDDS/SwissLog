import re
from tqdm import tqdm


class KnowledgeGroupLayer(object):

    def __init__(self, log_messages):
        self.log_messages = log_messages

    def run(self) -> list:
        for key, log in tqdm(self.log_messages.items(), desc='priori knowledge preprocess'):
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


            # date
            value = re.sub(
                r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', ' <*> ',
                value)
            # month
            value = re.sub(
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', ' <*> ',
                value)
            log['Content'] = value

        print('Knowledge group layer finished.')
        return self.log_messages
