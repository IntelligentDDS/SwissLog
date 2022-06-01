from tqdm import tqdm
from layers.layer import Layer
import pickle
from tqdm import tqdm
import wordninja

def hasDigit(inputString):
    return any(char.isdigit() for char in inputString)

def tolerant(source_dwords, target_dwords):
    rmt = target_dwords.copy()
    if len(source_dwords)<4: return False
    if len(target_dwords)<4: return False
    rms = set()
    for word in source_dwords:
        if word in target_dwords:
            rmt.remove(word)
        else:
            rms.add(word)
    return len(rmt)<=1 and len(rms) <=1


class DictOnlineGroupLayer(Layer):
    def __init__(self, dictionary_file=None, debug=False):
        self.dictionary = None
        self.debug = debug
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
        result = list()
        if self.debug:
            print("Getting wordset...\n")
        # for key, value in tqdm(self.log_messages.items(), desc='dictionaried'):
            # dictionary_words = set()
        dictionary_list = list()
        for word in log_entry['Content']:
            if hasDigit(word):
                continue
            word = word.strip('.:*')
            if word == '':
                continue
            if word in self.small_dict:
                dictionary_list.append(word)
            elif word in self.dictionary:
                # dictionary_words.add(word)
                dictionary_list.append(word)
                self.small_dict[word] = True
            elif all(char.isalpha() for char in word):
                if word in self.split_words_cache:
                    splitted_words = self.split_words_cache[word]
                else:
                    splitted_words = wordninja.split(word)
                    self.split_words_cache[word] = splitted_words
                for sword in splitted_words:
                    if len(sword) <= 2: continue
                    # dictionary_words.add(sword)
                    dictionary_list.append(sword)
        result_dict = dict(message=log_entry['Content'], dwords=dictionary_list, LineId=log_entry['LineId'])
        return result_dict

    def run(self, log_entry):
        return self.dictionaried(log_entry)

