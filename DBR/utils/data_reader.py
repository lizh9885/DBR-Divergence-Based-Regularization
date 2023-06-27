import json
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
import os
import numpy as np
import torch

mnli_label_dict = {
    "entailment":0,
    "neutral":1,
    "contradiction":2
}

hans_label_dict = {
    "entailment":0,
    "non-entailment":3
}

fever_label_dict = {
    "SUPPORTS":0,
    "REFUTES":1,
    "NOT ENOUGH INFO":2
}
class DataReader(object):
    """
        read data and turn into model feed form
    """
    def __init__(self, datasetname) -> None:
        self.datasetname = datasetname

    def get_examples(self, file_path):
        """
            load json data
        """
        with open(file_path, "r", encoding = "utf8") as fr:
            examples_list = json.load(fr)

        return examples_list

    
class NLIDataset(Dataset):
    def __init__(self, data, model_type, datasetname, max_len) -> None:
        super().__init__()
        self.data = data
        self.model_type = model_type
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", mirror = "tuna")
        self.datasetname = datasetname
        self.max_len = max_len
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        single_data = self.data[index]
        text_1 = single_data["text1"]
        text_2 = single_data["text2"]
        label = single_data["label"]
        if self.datasetname == "QQP":
            label = int(label)
        
        if self.datasetname == "hans":
            label = hans_label_dict[label]
        if self.datasetname == "fever":
            label = int(fever_label_dict[label])
        
        
        stich_text = text_1 + "[SEP]" + text_2
        single_data_token = self.tokenizer.tokenize(stich_text)[:self.max_len - 2]
        single_data_token = ["[CLS]"] + single_data_token + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(single_data_token)
        attention_mask = []
        segement_ids = []
        sep_flag = False
        for i in range(len(input_ids)):
            attention_mask.append(1)
                
            if input_ids[i] == 102 and not sep_flag:
                segement_ids.append(0)
                sep_flag = True
            elif sep_flag:
                segement_ids.append(1)
            else:
                segement_ids.append(0)
        assert len(attention_mask) == len(segement_ids) == len(input_ids)
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)
        segement_ids = np.array(segement_ids)
        length = len(input_ids)
        if len(single_data) == 3 or "pairID" in single_data:
            return input_ids, attention_mask, segement_ids, label, length
        else:# add index  
            topk_index = np.array(single_data["index"])
            return input_ids, attention_mask, segement_ids, label, length, topk_index
            


def recollate_fn(batch):
    input_ids, attention_mask, segement_ids, batch_label, batch_len = zip(*batch)
    
    cur_len_batch = len(batch)
    cur_max_len = max(batch_len)

    batch_input_ids = torch.LongTensor(cur_len_batch, cur_max_len).zero_()
    batch_attention_mask = torch.LongTensor(cur_len_batch, cur_max_len).zero_()
    batch_segement_ids = torch.LongTensor().new_ones(cur_len_batch, cur_max_len)

    for i in range(cur_len_batch):
        batch_input_ids[i,:batch_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_attention_mask[i,:batch_len[i]].copy_(torch.from_numpy(attention_mask[i]))
        batch_segement_ids[i,:batch_len[i]].copy_(torch.from_numpy(segement_ids[i]))

    batch_label = torch.LongTensor(batch_label)


    return {
        "input_ids":batch_input_ids,
        "attention_mask":batch_attention_mask,
        "segemnet_ids":batch_segement_ids,
        "label":batch_label
    }

def recollate_fn_v2(batch):
    input_ids, attention_mask, segement_ids, batch_label, batch_len, topk_index = zip(*batch)
    
    cur_len_batch = len(batch)
    cur_max_len = max(batch_len)
    
    grad_word_num = topk_index[0].shape[0]
    batch_input_ids = torch.LongTensor(cur_len_batch, cur_max_len).zero_()
    batch_attention_mask = torch.LongTensor(cur_len_batch, cur_max_len).zero_()
    batch_segement_ids = torch.LongTensor().new_ones(cur_len_batch, cur_max_len)
    batch_topk_index = torch.LongTensor(cur_len_batch, grad_word_num).zero_()
    
    for i in range(cur_len_batch):
        batch_input_ids[i,:batch_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_attention_mask[i,:batch_len[i]].copy_(torch.from_numpy(attention_mask[i]))
        batch_segement_ids[i,:batch_len[i]].copy_(torch.from_numpy(segement_ids[i]))
        batch_topk_index[i,:].copy_(torch.from_numpy(topk_index[i]))
    batch_label = torch.LongTensor(batch_label)


    return {
        "input_ids":batch_input_ids,
        "attention_mask":batch_attention_mask,
        "segemnet_ids":batch_segement_ids,
        "label":batch_label,
        "topk_index":batch_topk_index
    }
        


if __name__ == "__main__":
    reder = DataReader("mnli")
    json_data = reder.get_examples("../datasets/mnli/model_feed/train.json")[:11]

    NLIset = NLIDataset(json_data,"bert-base-cased","mnli",512)

    loader = DataLoader(dataset=NLIset,batch_size = 3,shuffle = True, collate_fn = recollate_fn)
    for i,batch in enumerate(loader):
        print(batch)
        
    