import argparse
import logging
import os
import torch
import random 
import pandas as pd
import numpy as np
from main import *
from evaluate import *
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datasetname",
                        type=str,
                        default="mnli",
                        help="training corpus name")

    parser.add_argument("--model_type",
                        type = str,
                        default = "bert-base-uncased",
                        help = "pretrained model used for training")

    parser.add_argument("--max_len",
                        type = int,
                        default = 512,
                        help = "the max length for tokenizer")

    parser.add_argument("--label_num",
                        type = int,
                        default = 3,
                        help = "label numbers default as 'entailment','netural' and 'contradiction'")

    parser.add_argument("--random_seed",
                        type = int,
                        default = 1234,
                        help = "random seed used for initalize parameter")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")      

    parser.add_argument("--evaluate",
                        action = 'store_true',
                        help = "whether to perform evaluation or train")
    
    parser.add_argument("--batch_size",
                        type = int,
                        default = 18 ,
                        help = "batch size used for training and evaluation")
    
    parser.add_argument("--train_epoch",
                        type = int,
                        default = 15,
                        help ="training epoch")

    parser.add_argument("--print_step",
                        type = int,
                        default = 500,
                        help = "loss logging step")

    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--dropout_prob",
                        default = 0.5,
                        type = float,
                        help = "dropout prob")

    parser.add_argument("--eps",
                        default= 1e-6,
                        type = float,
                        help = "The parameter of optimizer")
    
    parser.add_argument("--dev_epoch",
                        default = 3,
                        type = int,
                        help = "evaluate after per dev_epoch")
    
    parser.add_argument("--bias_word_num",
                        default = 3,
                        type = int,
                        help = "top k token feed to bias_model")

    parser.add_argument("--bias_middle_dim",
                        default = 100,
                        type = int,
                        help = "MLP middle dimension of bias_model")

    parser.add_argument("--use_product",
                        action = 'store_true',
                        help = "top k vector use product")

    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    #setting of log file
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level = logging.INFO)
    logger = logging.getLogger()
    if not os.path.exists('./log'):
        os.makedirs('./log')
    # f = logging.FileHandler('./log/mnli/final_train_2000_v2','w')
    # logger.addHandler(f)


    
    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    device = "cpu"
    n_gpu = torch.cuda.device_count()
    logger.info('device:{}, n_gpu:{}'.format(device, n_gpu))

    
    # seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    #result_dir and ckpt dir
    if not os.path.exists('./result'):
        os.makedirs('./result')
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    
    orign_model_path = "./ckpt/mnli/orign_model_epoch_6.pth"
    # orign_model_path = "./ckpt/fever/orign_model_epoch_7.pth"
    # orign_model_path = "./ckpt/QQP/orign_model_epoch_14.pth"
    
    bias_model_path =  "./ckpt/mnli/bias_model_2000.pth" #1.5jsd
    # bias_model_path = "./ckpt/fever/bias_model_3000.pth" 3jsd
    # bias_model_path = "./ckpt/QQP/bias_model_2000.pth"
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
        
    # evaluate_model(args, logger, model_path)
    # train_and_evaluate_final_model(args, logger, device, orign_model_path, bias_model_path)
    # train_and_evaluate_orign_model(args, logger, device)
    # train_and_evaluate_bias_head(args, logger, device, orign_model_path)
    
    

    # confidence curve draw
    # from evaluate import evaluate_model

    # model_path = "./ckpt/mnli/final_model_1000_epoch_7.pth"

    # all_acc_list, all_confidence_dict = evaluate_model(args, logger, model_path)

    # print(all_acc_list)
    # def writerCsv(res_dict):
    #     fileHeader = []
    #     max_len = 0
        
    #     data = [0] * len(res_dict)
    #     for key, value in res_dict.items():
    #         fileHeader.append(key)
    #         max_len = max(max_len, len(value))
    #     csvFile = open("confidence_final.csv", "w", newline='', encoding='utf8')
    #     writer = csv.writer(csvFile)
    #     writer.writerow(fileHeader)
    #     for i in range(max_len):
    #         for j in range(len(data)):
    #             if len(res_dict[fileHeader[j]]) > i:
    #                 data[j] = res_dict[fileHeader[j]][i] 
                    
    #             else:
    #                 data[j] = "None"
    #         writer.writerow(data)
        
        
    #     csvFile.close()
    # writerCsv(all_confidence_dict)