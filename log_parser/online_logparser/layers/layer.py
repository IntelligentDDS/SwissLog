import re
# import jieba.posseg as pseg
# import matplotlib.pyplot as plt
from tqdm import tqdm


class Layer(object):


    def evaluate(self, sim_hash_dict: dict):
        score_list = []
        for k in tqdm(sim_hash_dict.keys()):
            score = len(sim_hash_dict[k])
            flag = 0
            template = sim_hash_dict[k][0]['message']
            for obj in sim_hash_dict[k]:
                if flag != 0:
                    score = score - (1 - self.JaccardSim(template, obj['message']))
                flag = flag + 1
            final_score = score * 1.0 / len(sim_hash_dict[k])
            score_list.append(final_score)
        return score_list
