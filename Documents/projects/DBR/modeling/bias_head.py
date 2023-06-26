from torch import nn
import torch


class bias_model(nn.Module):
    def __init__(self,
                bias_word_num = 3,
                bias_middle_dim = 100,
                bert_dim = 768,
                dropout_prob = 0.5):
        super().__init__()
        # biased feed forawad layer
        self.bias_word_num = bias_word_num
        self.bias_middle_num = bias_middle_dim
        self.bert_dim = bert_dim
        self.dropout_prob = dropout_prob
        self.bias_project = nn.Sequential(nn.Linear(self.bert_dim * self.bias_word_num, 
                                                    self.bias_middle_num),nn.ReLU())
        self.bias_predict = nn.Linear(self.bias_middle_num, self.bias_word_num)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.sm = nn.Softmax(dim = 1)
        
    def forward(self, shortcut_data):
        # shortcut_data_size:(batch_size, bias_word_num * bert_dim)

        middle_project_head =self.dropout(self.bias_project(shortcut_data))#(batch_size, bias_middle_num)
        logits_head = self.bias_predict(middle_project_head)#(batch_size, label_num)

        prob_head = self.sm(logits_head)
        
        return logits_head, prob_head




        
        