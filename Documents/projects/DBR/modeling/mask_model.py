from torch import nn
from transformers import BertForSequenceClassification,BertModel
import torch
import numpy as np
class NLI_mask_model(nn.Module):
    def __init__(self,
                gpu = True,
                model_type = "bert-base-uncased",
                label_num = 3,
                train = True,
                
                ):
        super().__init__()
        self.use_gpu = gpu
        self.label_num = label_num
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=self.label_num,output_hidden_states = True)
        self.bert_dim = self.bert.config.hidden_size
        self.nli_head = nn.Linear(self.bert_dim,self.label_num)
        self.drop = nn.Dropout(0.5)
        self.train_mode = train
        self.sm = nn.Softmax(dim = 1)

        

    def load_dict(self,model_path):
        if self.use_gpu:
            sdict = torch.load(model_path)
            self.load_state_dict(sdict)
            self.to('cuda')

        else:
            sdict = torch.load(model_path)
            self.load_state_dict(sdict)
    
    def tensor_to_cuda(self, batch_data):
        if self.use_gpu:
            for key,value in batch_data.items():
                batch_data[key] = value.to("cuda")

        return batch_data

    def forward(self, batch_data, probs,copy_input_ids):
        batch_data = self.tensor_to_cuda(batch_data = batch_data)
        logits_bias = self.get_logits(input_ids = batch_data["input_ids"],
                                      attention_mask = batch_data["attention_mask"],
                                      token_type_ids = batch_data["segemnet_ids"])
        
        soft_mask_copy_input_ids = self.soft_mask(copy_input_ids, batch_data["topk_index"], probs)
        
        logits_debias = self.get_logits(input_ids = soft_mask_copy_input_ids,
                                        attention_mask = batch_data["attention_mask"],
                                        token_type_ids = batch_data["segemnet_ids"])
        
#         cls_pool_vec = self.bert(input_ids = batch_data["input_ids"],
#                                  attention_mask = batch_data["attention_mask"],
#                                  token_type_ids = batch_data["segemnet_ids"])[1]#(batch,hidden_size)
#         logits = self.nli_head(self.drop(cls_pool_vec))#(batch,label_num)   
#         probs = self.sm(logits)
#         return logits, probs       
        return logits_bias, logits_debias
    def all_mask(self, batch_data, copy_input_ids):
        batch_data = self.tensor_to_cuda(batch_data = batch_data)
        logits_bias = self.get_logits(input_ids = batch_data["input_ids"],
                                      attention_mask = batch_data["attention_mask"],
                                      token_type_ids = batch_data["segemnet_ids"])
        logits_debias = self.get_logits(input_ids = copy_input_ids,
                                        attention_mask = batch_data["attention_mask"],
                                        token_type_ids = batch_data["segemnet_ids"])
        return logits_bias, logits_debias
        
    def get_logits(self, input_ids, attention_mask, token_type_ids):
        """
            this function is used for IntergratedGradients
        """
        if self.use_gpu:
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            token_type_ids = token_type_ids.to("cuda")
        
        cls_pool_vec = self.bert(input_ids = input_ids,
                                attention_mask = attention_mask,
                                token_type_ids = token_type_ids)[1]     
        
        logits = self.nli_head(self.drop(cls_pool_vec)) #(batch, label_num)

        return logits

    def get_encode_text(self,batch_data):
        batch_data = self.tensor_to_cuda(batch_data = batch_data)
        encode_text = self.bert(input_ids = batch_data["input_ids"],
                                 attention_mask = batch_data["attention_mask"],
                                 token_type_ids = batch_data["segemnet_ids"])[0]#(batch,seq_len, hidden_size)

        return encode_text
    
    def soft_mask(self, input_ids, topk_index, probs):
        #probs 列表
        token_index = topk_index.to("cuda")#(batch, bias_word_num)
        with torch.no_grad():
            batch_size = input_ids.size()[0]
            batch_index = torch.arange(batch_size).reshape(batch_size,1).to("cuda")#(batch, 1)
        
        # all mask
#         input_ids[batch_index, token_index] = 103
        
        # asoft mask
        with torch.no_grad():
            bool_probs = np.array(np.random.binomial(1, probs), dtype = bool) #[True, False, False, True...]
            soft_batch_index = batch_index[bool_probs] #(soft_num_batch, 1)
            soft_token_index = token_index[bool_probs] #(soft_num_batch, bias_word_num)

        input_ids[soft_batch_index, soft_token_index] = 103
        
        return input_ids

    

    
        



