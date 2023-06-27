import torch
from torch import Tensor
from torch.nn.functional import softmax
from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BatchEncoding, PreTrainedTokenizer,BertModel, BertTokenizer
from utils.data_reader import NLIDataset, DataReader, recollate_fn
from torch.utils.data import DataLoader
from modeling.IntegratedGradient import int_grad, int_grad_v2
import json
from main import index_output
import sys
from modeling.model import NLImodel
import numpy as np
import os
from run import parse_args

if __name__ == "__main__":
    args = parse_args()
    



    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    n_gpu = torch.cuda.device_count()
    

    # orign_model_path = "./ckpt/mnli/orign_model_epoch_6.pth"
    # orign_model_path = "./ckpt/fever/orign_model_epoch_7.pth"
    orign_model_path = "./ckpt/QQP/orign_model_epoch_14.pth"
    #train_data is mnli train, if dev, dev data from mnli_dev
    reder = DataReader(args.datasetname)
    train_json_data = reder.get_examples("./datasets/" + args.datasetname + "/model_feed/train.json")
    total_train_example_len = len(train_json_data)
    print(total_train_example_len)

    # orignmodel output gradient vector
    orign_model = NLImodel(gpu = not args.no_cuda,
                     model_type = args.model_type,
                     label_num = args.label_num,
                     train = True
                    )
    orign_model.to(device)
    orign_model.eval()
    orign_model.load_state_dict(torch.load(orign_model_path))
    orign_model.zero_grad()
    print('model load')
    
    def cut_list(lists, cut_len):
        """
        将列表拆分为指定长度的多个列表
        :param lists: 初始列表
        :param cut_len: 每个列表的长度
        :return: 一个二维数组 [[x,x],[x,x]]
        """
        res_data = []
        if len(lists) > cut_len:
            for i in range(int(len(lists) / cut_len)):
                cut_a = lists[cut_len * i:cut_len * (i + 1)]
                res_data.append(cut_a)

            last_data = lists[int(len(lists) / cut_len) * cut_len:]
            if last_data:
                res_data.append(last_data)
        else:
            res_data.append(lists)

        return res_data


    cut_list_data = cut_list(train_json_data, 10000)
    print(len(cut_list_data))

    for i in range(len(cut_list_data)):
        try:
            cut_data = cut_list_data[i]
            index_output(args, cut_data, device, orign_model)    
            print("i {} sub list process over".format(i))    
        except RuntimeError as e:
            print("i {} runtime error".format(i))
            sys.exit()
    # check
    # cur_data = [
    #     {
    #     "text1": "One of our number will carry out your instructions minutely.",
    #     "text2": "A member of my team will execute your orders with immense precision.",
    #     "label": 0,
        
    #     },
    #     {
    #     "text1": "But a few Christian mosaics survive above the apse is the Virgin with the infant Jesus, with the Archangel Gabriel to the right (his companion Michael, to the left, has vanished save for a few feathers from his wings).",
    #     "text2": "Most of the Christian mosaics were destroyed by Muslims.  ",
    #     "label": 1,
    # }
    # ]
    # index_output(args, cur_data, device, orign_model)    