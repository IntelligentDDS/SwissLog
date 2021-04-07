# SwissLog
This repository is the basic implementation of our publication in ISSRE'20  conference paper [SwissLog: Robust and Unified Deep Learning Based Log Anomaly Detection for Diverse Faults](https://ieeexplore.ieee.org/abstract/document/9251078/). SwissLog contains two parts: log parsing and anomaly detection. We first open source the log parsing part here. 


## Description 
SwissLog adopts a novel log parsing method and extracts multiple templates by tokenizing, dictionarizing, and clustering history log data. Unlike other log parsing methods, our dictionary-based method requires no parameter tuning process. These templates are kept as natural sentences instead of event ids. We link those log statements with the same identifiers or simply use a sliding window to construct log sequences named sessions. And then the log sequence is transformed into semantic information and temporal information. SwissLog uses BERT encoder to encode semantic information into semantic embedding and projects temporal information onto time embedding. The concatenation of semantic embedding  and time embedding as input is fed into Attn-based Bi-LSTM to learn the features of normal, abnormal and performance-anomalous log sequence.

## Project Structure
The file structure is as belows:
```
└─log_parser
    ├─logs
    └─offline_logparser
        │  EngCorpus.pkl
        │  run.py
        │  
        ├─evaluator
        │  │  evaluator.py
        │  │  __init__.py
        │          
        └─layers
            │  dict_group_layer.py (for dictionarizing)
            │  file_output_layer.py (for file output)
            │  knowledge_layer.py (for preprocessing)
            │  mask_layer.py (for applying LCS and mergeing using prefix tree)
            │  tokenize_group_layer.py (for tokenizing)
            └─__init__.py
```

## Datasets
This demo adopts logpai benchmark. [Logpai](https://github.com/logpai/logparser) adopts 16 real-world log datasets ranging from distributed systems, supercomputers, operating systems, mobile systems, server applications, to standalone software including HDFS, Hadoop, Spark, Zookeeper, BGL, HPC, Thunderbird, Windows, Linux, Android, HealthApp, Apache, Proxifier, OpenSSH, OpenStack, and Mac. The above log datasets are provided by [LogHub](https://github.com/logpai/loghub). Each dataset contains 2,000 log samples with its ground truth tagged by a rule-based log parser.
## Requirements
This project can be reproducible under python v3.7. Please follow the command to install other key packages. 
```
pip install -r requirements.txt
```

## Quick Start
### Step 1: Construct a dictionary
We first construct a dictionary and utilize an English corpus including 5.2 million sentences, which is accessible on the [repository](https://github.com/brightmart/nlp_chinese_corpus) (or you can directly download this in this [link](https://storage.googleapis.com/nlp_chinese_corpus/translation2019zh.zip)). After splitting this corpus with the space delimiter, we collect 588,054 distinct words. Noting that not every occurred word is valid (e.g., location name), we set an occurrence threshold to filter common valid words. The dictionary finally remains only 18,653 common words. In the evaluation, we will use these 18,653 common words as the dictionary D to recognize valid words. The dictionary is stored as the file `EngCorpus.pkl`

It is also fine if you would like to use your own dictionary. Please carefully follow the dictionary format. For now, the program only receives the `.pkl` file storing the dict structure where the key is the word and the value is the occurrence. 

### Step 2: Just run the file
Please execute the `run.py` file in the offline_logparser directory. 
```
cd log_parser/offline_logparser
python3 run.py --dictionary=$PATH_OF_DICTIONARY
```



## Results
In this demo, we present benchmark results on 16 datasets. Overall, we observe that SwissLog shows almost the best PA in all datasets except the Mac logs. Even more, SwissLog can parse HDFS, BGL, Windows, Apache, OpenSSH datasets with 1.000 accuracy. The average of SwissLog is up to 0.962, which is much more than other log parsers by 10%. 

|dataset | F1_measure |Accuracy |
|---|----|---|
| HDFS | 1.000000   |   1.0000|
| Hadoop | 0.999901  |    0.9920 |
|  Spark | 0.999978   |   0.9965|
| Zookeeper | 0.999763  |    0.9845|
|  BGL | 0.999831     | 0.9695|
|   HPC | 0.992245     | 0.9095|
|Thunderbird  |0.999980 |     0.9920|
|  Windows | 1.000000    |  1.0000|
|      Linux | 0.989943  |    0.8690|
|   Andriod | 0.995815   |   0.9535|
| HealthApp | 0.993429   |   0.9010|
|   Apache | 1.000000   |   1.0000|
|  Proxifier | 0.999980 |     0.9900|
|OpenSSH | 1.000000     | 1.0000|
|  OpenStack | 1.000000 |     1.0000|
|     Mac  |0.976316    |  0.8400|
|Average |0.9967 |0.9623 | 


## Acknowledges:
SwissLog is implemented based on [LogPai team](https://github.com/logpai), we appreciate their contributions to the community. 

We also thank for all the contributors to this project:
|Name | github|
|---|---|
|Xiaoyun Li | @humanlee1011 |
|Pengfei Chen* | @chen0031 |
|Linxiao Jing | @jl0x61 |
|Zilong He | @QAZASDEDC |
|Guangba Yu |@yuxiaoba |


## Reference
Please cite our ISSRE'20 paper if you find this work is helpful. 

```
@inproceedings{li2020swisslog,
  title={SwissLog: Robust and Unified Deep Learning Based Log Anomaly Detection for Diverse Faults},
  author={Li, Xiaoyun and Chen, Pengfei and Jing, Linxiao and He, Zilong and Yu, Guangba},
  booktitle={2020 IEEE 31st International Symposium on Software Reliability Engineering (ISSRE)},
  pages={92--103},
  year={2020},
  organization={IEEE}
}
```

