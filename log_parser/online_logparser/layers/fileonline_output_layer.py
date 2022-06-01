from layers.layer import Layer
import os
import hashlib
import csv


class FileOnlineOutputLayer(Layer):
    def __init__(self, log_messages,  results: dict, filename: str, templates: list, message_headers: list):
        self.log_messages = log_messages
        self.filename = filename
        self.results = results
        self.templates = templates
        self.message_headers = message_headers

    def output_csv(self, filename, messages, headers):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for key, row in messages.items():
                writer.writerow(row)


    def outputResult(self):
        # import pdb; pdb.set_trace()
        log_events = dict()     
        eids = dict()
        for idx, val in self.results.items():
            temp = ' '.join(self.templates[val])
            if temp not in eids:
                eids[temp] = hashlib.md5(temp.encode('utf-8')).hexdigest()[0:8]
            self.log_messages[idx]['EventTemplate'] = temp

            self.log_messages[idx]['EventTemplate'] = temp
            self.log_messages[idx]['EventId'] = eids[temp]
        tot = 0
        for temp, eid in eids.items():
            log_events[tot] = dict(EventId=eid, EventTemplate=temp)
            tot += 1
   

        self.message_headers += ['EventId', 'EventTemplate']
        event_headers = ['EventId', 'EventTemplate']

        self.output_csv(self.filename+'_structured.csv', self.log_messages, self.message_headers)
        self.output_csv(self.filename+'_templates.csv', log_events, event_headers)

    def run(self):
        dirname = os.path.dirname(self.filename)
        os.makedirs(dirname, exist_ok=True)
        self.outputResult()