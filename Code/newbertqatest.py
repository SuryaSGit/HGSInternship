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
for i in range(len(cutdata)):
    paragraph += cutdata[i]


#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#new_tokenizer = tokenizer.train_new_from_iterator(cutdata, vocab_size=25000)
#new_tokenizer.save_pretrained("my-new-tokenizer")
def convert_ids_to_string(tokenizer, input_ids):
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids)
    )

device = "cuda"
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',use_fast=False)
model = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad", return_dict=False
).to(device)
question = "What do they call petrol in US?"


def chunkify(question, paragraph):
    inputs = tokenizer.encode_plus(question, paragraph, return_tensors="pt").to(device)
    qmask = inputs["token_type_ids"].lt(1)
    qt = torch.masked_select(inputs["input_ids"], qmask)
    chunk_size = model.config.max_position_embeddings - qt.size()[0] - 1
    chunked_input = OrderedDict()
    for k, v in inputs.items():
        q = torch.masked_select(v, qmask)
        c = torch.masked_select(v, ~qmask)
        chunks = torch.split(c, chunk_size)

        for i, chunk in enumerate(chunks):
            if i not in chunked_input:
                chunked_input[i] = {}

            thing = torch.cat((q, chunk))
            if i != len(chunks) - 1:
                if k == "input_ids":
                    thing = torch.cat((thing, torch.tensor([102])))
                else:
                    thing = torch.cat((thing, torch.tensor([1])))

            chunked_input[i][k] = torch.unsqueeze(thing, dim=0)

    return chunked_input


def answer(question, chunked_input):
    answer = ""
    max = 0
    # now we iterate over our chunks, looking for the best answer from each chunk
    for _, chunk in chunked_input.items():
        answer_start_scores, answer_end_scores = model(**chunk)
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        testasf = torch.max(answer_start_scores)
        ans = convert_ids_to_string(
            tokenizer, chunk["input_ids"][0][answer_start:answer_end]
        )
        # if the ans == [CLS] then the model did not find a real answer in this chunk
        if (
            "[CLS]" not in ans
            and "[SEP]" not in ans
            and len(ans) < 150
            and max < testasf.item()
        ):
            max = testasf.item()
            answer = ans
    print(answer)
    print(f'Max {max}')
    return answer
chunked_input = chunkify(question, paragraph)
corrected_answer = answer(question, chunked_input)
print("Human: ", question)
print("BERT: ", corrected_answer)
end = time.time()
print(end - start_time)
