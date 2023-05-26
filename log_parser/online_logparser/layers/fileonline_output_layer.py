from layers.layer import Layer
import os
import hashlib
import csv
import re


class FileOnlineOutputLayer(Layer):
    def __init__(self, log_messages,  results: dict, filename: str, templates: list, message_headers: list, keep_para):
        self.log_messages = log_messages
        self.filename = filename
        self.results = results
        self.templates = templates
        self.message_headers = message_headers
        self.keep_para = keep_para

    def output_csv(self, filename, messages, headers):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for key, row in messages.items():
                writer.writerow(row)


    def outputResult(self):
        log_events = dict()     
        eids = dict()
        for idx, val in self.results.items():
            # fix bug: don't use spaces to join all tokens in the template
            temp = ''.join(self.templates[val])
            temp = " ".join(temp.strip().split())
            if temp not in eids:
                eids[temp] = hashlib.md5(temp.encode('utf-8')).hexdigest()[0:8]
            self.log_messages[idx]['EventTemplate'] = temp

            self.log_messages[idx]['EventTemplate'] = temp
            self.log_messages[idx]['EventId'] = eids[temp]
            if self.keep_para:
                self.log_messages[idx]["ParameterList"] = []
                self.log_messages[idx]["ParameterList"] = self.get_parameter_list(self.log_messages[idx])
        tot = 0
        for temp, eid in eids.items():
            log_events[tot] = dict(EventId=eid, EventTemplate=temp)
            tot += 1
   

        self.message_headers += ['EventId', 'EventTemplate', 'ParameterList', 'Message']
        event_headers = ['EventId', 'EventTemplate']

        self.output_csv(self.filename+'_structured.csv', self.log_messages, self.message_headers)
        self.output_csv(self.filename+'_templates.csv', log_events, event_headers)

    def get_parameter_list(self, messages):
        # import pdb; pdb.set_trace()
        # template_regex = re.sub(r"<.{1,5}>", "<*>", messages["EventTemplate"])
        template_regex = messages["EventTemplate"]
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        compiled_regex = re.compile(template_regex)
        parameter_list = compiled_regex.findall("".join(messages["Message"]))
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list
    

    def run(self):
        dirname = os.path.dirname(self.filename)
        os.makedirs(dirname, exist_ok=True)
        self.outputResult()