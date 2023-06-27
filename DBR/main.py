import logging
import os
import torch
import sys
import random 
from torch.utils.data import DataLoader
from modeling.model import NLImodel
from modeling.bias_head import bias_model
from modeling.mask_model import NLI_mask_model
from transformers import BertTokenizer
import numpy as np
from torch import nn
from transformers.optimization import AdamW
from utils.data_reader import NLIDataset, DataReader, recollate_fn, recollate_fn_v2
import torch.optim as optim
from modeling.IntegratedGradient import int_grad,int_grad_v2
import torch.nn.functional as F
from utils.loss_fc import rewighting_loss, JSD, cal_loss, cal_var
import csv
import json

def train_and_evaluate_orign_model(args, logger, device):
    """
        this function trains a orign model: step 1
    """
    # load data
    reder = DataReader(args.datasetname)
    train_json_data = reder.get_examples("./datasets/" + args.datasetname + "/model_feed/train.json")
    total_train_example_len = len(train_json_data)

    train_NLIset = NLIDataset(train_json_data,args.model_type,args.datasetname,args.max_len)

    train_loader = DataLoader(dataset = train_NLIset, 
                                batch_size = args.batch_size,
                                shuffle = True,
                                collate_fn = recollate_fn)
    logger.info("*** train data loaded ***")
    
    # prepare model
    model = NLImodel(gpu = not args.no_cuda,
                     model_type = args.model_type,
                     label_num = args.label_num,
                     train = True
                    )
    model.to(device)
    logger.info("*** model loaded ***")
    

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr = args.learning_rate,
                      eps = args.eps,
                      correct_bias = False)


    loss_fn = nn.CrossEntropyLoss()
    logger.info("*** Running Training ***")
    logger.info("*** all training examples = {}".format(total_train_example_len))
    logger.info("*** training epoches = {}".format(args.train_epoch))
    logger.info("*** Num steps per epoch = {}".format(len(train_loader)))
    best_acc = 0
    for epoch in range(args.train_epoch):
        model.train()
        logging_loss = 0
        logger.info("*** epoch {} / {} ***".format(epoch + 1, args.train_epoch))
        for i,batch in enumerate(train_loader):
            
            logits, probs = model(batch_data = batch)
            loss_step = loss_fn(logits, batch["label"]) 
            logging_loss += loss_step.item()
            if (i+1) % args.print_step == 0:
                print_loss = logging_loss/args.print_step
                logger.info("step {} average loss : {:.4f}".format(i + 1, print_loss))
                logging_loss = 0
            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
        acc_list, confience = evaluate_orign(args, logger, model)
        
        logger.info(acc_list)
        

        torch.save(model.state_dict(), "./ckpt/QQP/orign_model_epoch_{}.pth".format(epoch + 1))
        
        # if (epoch + 1) % args.dev_epoch == 0:
        #     mean_acc = evaluate_orign(args, logger, model)
        #     if mean_acc >= best_acc:
        #         logger.info("save the orign model, best_acc {}".format(mean_acc))
        #         torch.save(bias_model.state_dict(), "/ckpt/orign_model.pth")
        #         best_acc = mean_acc


            
def index_output(args, cut_data, device, orign_model):
    grad_dataset = NLIDataset(cut_data,args.model_type,args.datasetname,args.max_len)
    
    grad_loader = DataLoader(dataset = grad_dataset, 
                             batch_size = args.batch_size, 
                             shuffle = False, 
                             pin_memory = False, 
                             num_workers = 8,
                             collate_fn = recollate_fn)
    data_numbers = len(cut_data)
    index_result = []
    num = 0 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for i, batch in enumerate(grad_loader):
        with torch.no_grad():
            for key,value in batch.items():
                 batch[key] = value.to(device)
            
            top_k_indices = int_grad_v2(args, orign_model, batch, tokenizer) #得到topk归因向量 #(batch, topk)
            # check
            # print(top_k_indices)
            # return top_k_indices
            if i % 1000 == 0:
                print("process {} examples".format((i+1)*args.batch_size))
        for j in range(top_k_indices.shape[0]):
            index_result.append(top_k_indices[j])

    assert data_numbers == len(index_result)   

    for i in range(data_numbers):
        cut_data[i].update({"index":index_result[i].tolist()})
    if not os.path.exists("./datasets/QQP/model_feed/train_withindex.json"):
        with open("./datasets/QQP/model_feed/train_withindex.json","a+" ,encoding = "utf8") as f:
            f.write(json.dumps(cut_data, ensure_ascii = False, indent = 4) + "\n")
            f.close()
    else:
        with open("./datasets/QQP/model_feed/train_withindex.json","r", encoding = "utf8") as fw:
            previous_data = json.load(fw)
            fw.close()
        previous_data.extend(cut_data)
        with open("./datasets/QQP/model_feed/train_withindex.json","w" ,encoding = "utf8") as f:
            f.write(json.dumps(previous_data, ensure_ascii = False, indent = 4) + "\n")
            f.close()
def train_and_evaluate_bias_head(args, logger, device, orign_model_path):
    """
        this function trains a bias head
    """
    #train_data is mnli train, if dev, dev data from mnli_dev
     reder = DataReader(args.datasetname)
    train_data = reder.get_examples("./datasets/" + args.datasetname + "/model_feed/train_withindex.json")
    test_data = reder.get_examples("./datasets/" + args.datasetname + "/model_feed/train_withindex.json")
    sample_len = 3000
    logger.info("randdom choose {} samples from total train examples".format(sample_len))
    sample_json_data = random.sample(train_data, sample_len)
    test_sample_json_data = random.sample(test_data, 10000)
    test_sample_json_data_new = [ele for ele in test_sample_json_data if ele not in sample_json_data]
    # orignmodel output gradient vector
    
    
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
    # print("orign model",torch.cuda.memory_allocated())
    # update train_json_data append topk vec
    # train_json_data = cal_topk_word(args, device, orign_model, train_json_data)
   
    train_NLIset = NLIDataset(sample_json_data,args.model_type,args.datasetname,args.max_len)
    
    train_loader = DataLoader(dataset = train_NLIset, 
                                batch_size = args.batch_size,
                                shuffle = True,
                                collate_fn = recollate_fn_v2)
    
    # train bias_head to obtain an extreme bias model
    bias_head_model = bias_model(bias_word_num = args.bias_word_num,
                            bias_middle_dim = args.bias_middle_dim,
                            bert_dim = orign_model.bert_dim,
                            dropout_prob = args.dropout_prob)

    bias_head_model.to(device)
    
    print("bias head", torch.cuda.memory_allocated())
    
    #prepare optimizer and loss_function
    optimizer = optim.Adam(bias_head_model.parameters(), 
                            lr = args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    logger.info("*** Running Training ***")
    logger.info("*** all training examples = {}".format(sample_len))
    logger.info("*** training epoches = {}".format(args.train_epoch))
    logger.info("*** Num steps per epoch = {}".format(len(train_loader)))
    best_acc = 0
    for epoch in range(1):
        bias_head_model.train()
        logging_loss = 0
        logger.info("*** epoch {} / {} ***".format(epoch + 1, args.train_epoch))
        for i, batch in enumerate(train_loader): 
            for key,value in batch.items():
                batch[key] = value.to(device)
            
            topk_index = batch["topk_index"]#(batch, bias_word_num)
            batch_size = topk_index.size()[0]
            batch_index = torch.arange(batch_size).reshape(batch_size,1).to(device)
            
            token_index = topk_index
            
            
            batch_vec = orign_model.get_encode_text(batch)#(batch, seq_len, bert_dim)
            index_vec = batch_vec[batch_index, token_index, :].view(batch_size, -1)#(batch, bias_word_num * bert_dim)

            logits_head, _ = bias_head_model(index_vec)
            
            
            loss_step = loss_fn(logits_head, batch["label"]) 
            logging_loss += loss_step.item()
            if (i+1) % args.print_step == 0:
                print_loss = logging_loss/args.print_step
                logger.info("step {} average loss : {:.4f}".format(i + 1, print_loss))
                logging_loss = 0
            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
        evaluate_bias(args, logger, bias_head_model, orign_model, test_sample_json_data_new)
        torch.save(bias_head_model.state_dict(), "./ckpt/mnli/bias_model_{}.pth".format(sample_len))
       
        
def evaluate_bias(args, logger, model, orign_model, test_sample_data):
    """
        bias model evaluation on train set , confidence 
    """

    
    
    logger.info("evaluating bias head model")
    dev_NLIset = NLIDataset(test_sample_data,args.model_type,args.datasetname,args.max_len)
    
    dev_loader = DataLoader(dataset = dev_NLIset, 
                                batch_size = args.batch_size,
                                shuffle = False,
                                collate_fn = recollate_fn_v2)
    
    model.eval()
    model.zero_grad()
    gold_num = 0
    correct_num = 0
    print_list = []
    with torch.no_grad():
        for step, batch in enumerate(dev_loader):
            topk_index = batch["topk_index"]#(batch, bias_word_num)
            batch_size = topk_index.size()[0]
            batch_index = torch.arange(batch_size).reshape(batch_size,1).to("cuda")
            token_index = topk_index
            batch_vec = orign_model.get_encode_text(batch)#(batch, seq_len, bert_dim)
            index_vec = batch_vec[batch_index, token_index, :].view(batch_size, -1)#(batch, bias_word_num * bert_dim)
            _, probs = model(index_vec)
            
            print_list.extend(probs.tolist())
            gold_num += probs.size()[0]
            
            predict = probs.argmax(dim = -1) #(batch)
            predict = predict.detach().cpu().numpy()
            label = batch["label"].cpu().numpy()
            correct_num += np.sum(label == predict)
            
    logger.info("correct_num {} gold_num {} acc {}".format(correct_num, gold_num, correct_num/gold_num))
   
        
        
def train_and_evaluate_final_model(args, logger, device, orign_model_path, bias_model_path):
    """
        final de bias model through bias head
    """

    # loda data
    reder = DataReader(args.datasetname)
    train_json_data = reder.get_examples("./datasets/" + args.datasetname + "/model_feed/train_withindex.json")
    total_train_example_len = len(train_json_data)

    train_NLIset = NLIDataset(train_json_data,args.model_type,args.datasetname,args.max_len)

    train_loader = DataLoader(dataset = train_NLIset, 
                                batch_size = args.batch_size,
                                shuffle = True,
                                collate_fn = recollate_fn_v2)
    logger.info("*** train data loaded ***")    

    # load orign model                    
    
    orign_model = NLImodel(gpu = not args.no_cuda,
                     model_type = args.model_type,
                     label_num = args.label_num,
                     train = True
                    )
    orign_model.load_state_dict(torch.load(orign_model_path))
    orign_model.to(device)
    orign_model.eval()
    orign_model.zero_grad()
    for name, param in orign_model.named_parameters():
        param.requires_grad = False
    logger.info("*** orign model loaded***")

#     # load bias head model
    bias_head_model = bias_model(bias_word_num = args.bias_word_num,
                            bias_middle_dim = args.bias_middle_dim,
                            bert_dim = orign_model.bert_dim,
                            dropout_prob = args.dropout_prob)

    bias_head_model.to(device)
    bias_head_model.load_state_dict(torch.load(bias_model_path))
    bias_head_model.eval()
    bias_head_model.zero_grad()
    for name, param in bias_head_model.named_parameters():
        param.requires_grad = False
    logger.info("*** bias head model loaded")
    
    # load final model
    final_model = NLI_mask_model(
                    gpu = not args.no_cuda,
                    model_type = args.model_type,
                    label_num = args.label_num,
                    train = True)

    final_model.to(device)
    final_model.train()
    
    logger.info("*** final model loaded")

    # Prepare optimizer
    param_optimizer = list(final_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr = args.learning_rate,
                      eps = args.eps,
                      correct_bias = False)

    # Running training
    
    loss_fn = nn.CrossEntropyLoss(reduction = "none") #为了后续加权
    cross_fn = nn.CrossEntropyLoss()
    loss_fn_product = nn.NLLLoss()
#     loss_fn_product = nn.CrossEntropyLoss()
    JSD_model = JSD()
    l_s = nn.LogSoftmax(dim = -1)
#     n_s = nn.Softmax(dim = -1)
    logger.info("*** Running Training ***")
    logger.info("*** all training examples = {} ***".format(total_train_example_len))
    logger.info("*** training epoches = {} ***".format(args.train_epoch))
    logger.info("*** Num steps per epoch = {} ***".format(len(train_loader)))
    best_acc = 0

    for epoch in range(args.train_epoch):
        final_model.train()
        logging_loss = 0
        logger.info("*** epoch {} / {} ***".format(epoch + 1, args.train_epoch))
        for i, batch in enumerate(train_loader):
            
            # way 2:Product-of-experts (PoE)
#             with torch.no_grad():
#             topk_index = batch["topk_index"].to(device)#(batch, bias_word_num)
#             batch_index = torch.arange(batch_size).reshape(batch_size,1).to(device)
            
#             batch_vec = orign_model.get_encode_text(batch)#(batch, seq_len, bert_dim)
#             index_vec = batch_vec[batch_index, topk_index, :].view(batch_size, -1)#(batch, bias_word_num * bert_dim)
#             logits_bias, probs_bias = bias_head_model(index_vec)
#             logits_debias, probs_debias = final_model(batch_data = batch)

            
            # way 2:Product-of-experts (PoE)
#             pt = F.softmax(logits_debias, dim = 1)
#             pt_adv = F.softmax(logits_bias, dim = 1)
#             joint_pt = F.softmax((torch.log(pt) + torch.log(pt_adv)), dim=1)
#             target = batch["label"].view(-1,1)
#             joint_p = joint_pt.gather(1, target)
#             batch_loss = -torch.log(joint_p)
#             cross_loss_product = batch_loss.mean()

            # way 3:JSD loss
            batch_size = batch["input_ids"].size()[0]
            with torch.no_grad():
                topk_index = batch["topk_index"].to(device)#(batch, bias_word_num)
                batch_index = torch.arange(batch_size).reshape(batch_size,1).to(device)
            batch_vec = orign_model.get_encode_text(batch)#(batch, seq_len, bert_dim)
            index_vec = batch_vec[batch_index, topk_index, :].view(batch_size, -1)#(batch, bias_word_num * bert_dim)
            copy_input_ids = batch["input_ids"].clone()
            # # soft mask
            _, probs_bias = bias_head_model(index_vec) #(batch, 3)
            var_probs = cal_var(probs_bias)
            
            logits_bias, logits_debias = final_model(batch, var_probs, copy_input_ids)

            #hard mask
            # copy_input_ids[batch_index, topk_index] = 103
            # logits_bias, logits_debias = final_model.all_mask(batch,copy_input_ids)


            # way 3: JSD loss
            jsd_loss = JSD_model.cal(logits_bias, logits_debias)
            cross_loss = cross_fn(logits_bias, batch["label"])
            
            sum_loss = 1 * cross_loss + 3 * jsd_loss

         
            logging_loss += sum_loss.item()
            if (i+1) % args.print_step == 0:
                print_loss = logging_loss/args.print_step
                logger.info("step {} average loss : {:.4f}".format(i + 1, print_loss))
                
                logging_loss = 0
            optimizer.zero_grad()
            sum_loss.backward()
            optimizer.step()
            
        acc_list, confience = evaluate_final(args, logger, final_model)
        logger.info(acc_list)


            



            
            
def evaluate_orign_subset(val_list, logger, dataname, args, model):
    acc_list = []#每个subset的accuracy
    confidence_dict = {}
    
    for ele in val_list:
        confidence_list = []
        val_len = len(ele)
        logger.info("*** evaluating {}, data len {}".format(dataname, val_len))
        
        

        eval_NLIset = NLIDataset(ele, args.model_type, dataname ,args.max_len)

        eval_loader = DataLoader(dataset = eval_NLIset, 
                                    batch_size = args.batch_size,
                                    shuffle = True,
                                    collate_fn = recollate_fn)
        correct_num = 0
        gold_num = 0
        
        with torch.no_grad():
            for step, batch in enumerate(eval_loader):
                
                logits = model.get_logits(batch["input_ids"], batch["attention_mask"], batch["segemnet_ids"])
                probs = F.softmax(logits, dim = -1)
                # 写入confiednce list
                row_index = torch.Tensor([i for i in range(batch["label"].size()[0])]).long()
                column_index = torch.Tensor(batch["label"].detach().cpu().numpy()).long()
#                 confidence = probs[row_index, column_index].detach().cpu().numpy().tolist()#列表形式的confidence分数
#                 confidence_list.extend(confidence)

                gold_num += probs.size()[0]
                predict = probs.argmax(dim = -1) #(batch)
                predict = predict.detach().cpu().numpy()
                confidence_list.extend(predict.tolist())
                label = batch["label"].cpu().numpy()
                
                
                if dataname == "hans":
                    
                    predict[predict == 1] = 3
                    predict[predict == 2] = 3
                             
                   
                correct_num += np.sum(label == predict)
                    
                if (step + 1) % 100 == 0:
                    logger.info("{} samples evaluated".format((step + 1) * args.batch_size))
        logger.info("dataset {}, subset_len {}, correct num {}, gold num {}, accuarcy {:.4f}".format(dataname, val_len, correct_num, gold_num, correct_num/gold_num))
        acc_list.append(correct_num/gold_num) #列表，每个元素为每个subset的accuarcy
        confidence_dict.update({"hans":confidence_list}) #字典，每个元素为当前subset的confidence_list
    return acc_list, confidence_dict
        


    

def evaluate_orign(args, logger, orign_model):
    """
        orign model evaluation 
    """
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    orign_model.eval()
    orign_model.zero_grad()
    all_acc_list = []
    all_confidence_dict = {}
    logger.info("evaluating orign model")
    # eval_list = ["mnli", "hans","mnli_hard",]
    # eval_list = ["fever"]
    eval_list = ["QQP"]
    for dataname in eval_list:
        reder = DataReader(datasetname = dataname)
        if dataname == "mnli":
            # 0,1,2    
            val_match_json_data = reder.get_examples("./datasets/mnli/model_feed/val_match.json" )
            val_mismatch_json_data = reder.get_examples("./datasets/mnli/model_feed/val_mismatch.json")

            val_list = [val_match_json_data, val_mismatch_json_data]
            
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, orign_model)
            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
        if dataname == "mnli_hard":
            # entailment -> 0
            dev_match_easy_json_data = reder.get_examples("./datasets/mnli_hard/model_feed/dev_matched_easy.json")
            dev_mismatch_easy_json_data = reder.get_examples("./datasets/mnli_hard/model_feed/dev_mismatched_easy.json")
            dev_match_hard_json_data = reder.get_examples("./datasets/mnli_hard/model_feed/dev_matched_hard.json")
            dev_mismatch_hard_json_data = reder.get_examples("./datasets/mnli_hard/model_feed/dev_mismatched_hard.json")
            val_list = [dev_match_easy_json_data, dev_match_hard_json_data, dev_mismatch_easy_json_data, dev_mismatch_hard_json_data]
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, orign_model)

            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
        if dataname == "hans":
            # entailment and no-entailment -> 0 and 3
            test_json_data = reder.get_examples("./datasets/hans/model_feed/test.json")
            val_list = [test_json_data]
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, orign_model)
            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
        if dataname == "fever":
            dev_json_data = reder.get_examples("./datasets/fever/model_feed/dev.json")
            v1_json_data = reder.get_examples("./datasets/fever/model_feed/v1.json")
            v2_json_data = reder.get_examples("./datasets/fever/model_feed/v2.json")
            val_list = [dev_json_data, v1_json_data, v2_json_data]
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, orign_model)
            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
        if dataname == "QQP":
            dev_json_data = reder.get_examples("./datasets/QQP/model_feed/dev.json")
            paws_json_data = reder.get_examples("./datasets/QQP/model_feed/paws.json")
            val_list = [dev_json_data, paws_json_data]
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, orign_model)
            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
    return all_acc_list, all_confidence_dict
    



def evaluate_final(args, logger, final_model):
    """
        fianl model evaluation only in domain to train a most bias model
    """
   
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    final_model.eval()
    final_model.zero_grad()
    all_acc_list = []
    all_confidence_dict = {}
    logger.info("evaluating orign model")
    eval_list = ["mnli", "hans","mnli_hard",]
    # eval_list = ["fever"]
    # eval_list = ["QQP"]
    for dataname in eval_list:
        reder = DataReader(datasetname = dataname)
        if dataname == "mnli":
            # 0,1,2    
            val_match_json_data = reder.get_examples("./datasets/mnli/model_feed/val_match.json" )
            val_mismatch_json_data = reder.get_examples("./datasets/mnli/model_feed/val_mismatch.json")

            val_list = [val_match_json_data, val_mismatch_json_data]
            
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, final_model)
            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
        if dataname == "mnli_hard":
            # entailment -> 0
            dev_match_easy_json_data = reder.get_examples("./datasets/mnli_hard/model_feed/dev_matched_easy.json")
            dev_mismatch_easy_json_data = reder.get_examples("./datasets/mnli_hard/model_feed/dev_mismatched_easy.json")
            dev_match_hard_json_data = reder.get_examples("./datasets/mnli_hard/model_feed/dev_matched_hard.json")
            dev_mismatch_hard_json_data = reder.get_examples("./datasets/mnli_hard/model_feed/dev_mismatched_hard.json")
            val_list = [dev_match_easy_json_data, dev_match_hard_json_data, dev_mismatch_easy_json_data, dev_mismatch_hard_json_data]
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, final_model)

            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
        if dataname == "hans":
            # entailment and no-entailment -> 0 and 3
            test_json_data = reder.get_examples("./datasets/hans/model_feed/test.json")
            val_list = [test_json_data]
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, final_model)
            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
        if dataname == "fever":
            dev_json_data = reder.get_examples("./datasets/fever/model_feed/dev.json")
            v1_json_data = reder.get_examples("./datasets/fever/model_feed/v1.json")
            v2_json_data = reder.get_examples("./datasets/fever/model_feed/v2.json")
            val_list = [dev_json_data, v1_json_data, v2_json_data]
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, final_model)
            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
        if dataname == "QQP":
            dev_json_data = reder.get_examples("./datasets/QQP/model_feed/dev.json")
            paws_json_data = reder.get_examples("./datasets/QQP/model_feed/paws.json")
            val_list = [dev_json_data, paws_json_data]
            acc_list, confidence_dict = evaluate_orign_subset(val_list, logger, dataname, args, final_model)
            all_acc_list.extend(acc_list)
            all_confidence_dict.update(confidence_dict)
    return all_acc_list, all_confidence_dict
    
                        

    


if __name__ == "__main__":
    pass
   
            # way 1: Example reweighting (ER) 
#             cross_loss_weight = rewighting_loss(loss_func = loss_fn,
#                                                 logits = logits_debias,
#                                                 label = batch["label"],
#                                                 weight = probs_bias)
#             cross_loss_weight = loss_fn_product(logits_debias, batch["label"])