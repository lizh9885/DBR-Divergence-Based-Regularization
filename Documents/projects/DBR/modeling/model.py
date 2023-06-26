from torch import nn
from transformers import BertModel, BertForSequenceClassification
import torch
class NLImodel(nn.Module):
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

    def forward(self, batch_data):
        batch_data = self.tensor_to_cuda(batch_data = batch_data)

        cls_pool_vec = self.bert(input_ids = batch_data["input_ids"],
                                 attention_mask = batch_data["attention_mask"],
                                 token_type_ids = batch_data["segemnet_ids"])[1]#(batch,hidden_size)
        logits = self.nli_head(self.drop(cls_pool_vec))
        probs = self.sm(logits)
        return logits, probs           

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
        logits = self.nli_head(self.drop(cls_pool_vec))
         #(batch, label_num)

        return logits

    def get_encode_text(self,batch_data):
        batch_data = self.tensor_to_cuda(batch_data = batch_data)
        encode_text = self.bert(input_ids = batch_data["input_ids"],
                                 attention_mask = batch_data["attention_mask"],
                                 token_type_ids = batch_data["segemnet_ids"])[0]#(batch,seq_len, hidden_size)

        return encode_text

    

    

    
        



