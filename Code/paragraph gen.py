import json
import html
import unicodedata
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AutoTokenizer
from collections import OrderedDict
import time

start_time = time.time()

with open("EasyDNNNews_202302161844.json") as json_file:
    data = json.load(json_file)
trimmed = []
for i in range(len(data)):
    newdata = data[i]
    if newdata["PortalID"] == 6:
        trimmed.append(newdata)
cutdata = []
for i in range(len(trimmed)):
    newtrimmed = trimmed[i]
    text = newtrimmed["CleanArticleData"]
    cleantext = html.unescape(text)
    cleanertext = unicodedata.normalize("NFKD", cleantext)
    cutdata.append(cleanertext)
paragraph = ""
with open('test.txt', 'w') as f:
    f.write(cutdata[0])
for i in range(1,len(cutdata)):
    with open('test'+str(i)+'.txt', 'w') as f:
        f.write(cutdata[i])
    paragraph += cutdata[i]

