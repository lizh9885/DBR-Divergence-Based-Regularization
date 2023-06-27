import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from captum.attr import LayerIntegratedGradients



def int_grad(args, model, batch_data):
    """
        this function is used for obataing the top k stack vector, including following steps:
            step 1: prepare for attribute function inputs, especially ref_input_ids
            step 2: calculate the top k gradient vector, 
                    this includes two ways: l2 norm or product of bert encode vector and gradient vector
            step 3: flat this top k vector and output it
        parameters: args
                    batch_data: inputs
                    model: orign_model
        return: flatten vector

    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def predict(input_ids, attention_mask, token_type_ids):
        return model.get_logits(input_ids, attention_mask, token_type_ids)
    # step 1: prepare for attribution inputs
    lig = LayerIntegratedGradients(predict, model.bert.embeddings)
    
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
    device = "cuda"
    input_ids = batch_data["input_ids"].to(device)#(batch, seq_len)
    batch_data["label"] = batch_data["label"].to(device)
    batch_data["attention_mask"] = batch_data["attention_mask"].to(device)
    batch_data["segemnet_ids"] = batch_data["segemnet_ids"].to(device)
    batch_size, seq_len = input_ids.size()
    ref_input_ids = input_ids.clone()
    ref_input_ids[(ref_input_ids != tokenizer.cls_token_id)*(ref_input_ids != tokenizer.sep_token_id)] = tokenizer.pad_token_id
    # sep_index = torch.nonzero(input_ids == sep_token_id)#(batch * 2, 2)
    # assert sep_index.size()[0] == input_ids.size()[0] * 2
    # first_sep_index_list = list(range(0, sep_index.size()[0], 2)) #选出text1末尾的sep_token_id
    # sep_index_first = sep_index.detach().cpu().numpy()[first_sep_index_list]#(batch,2) 2 代表(batch 中的第几个句子，middle_sep_index)
    # second_sep_index_list = list(range(1, sep_index.size()[0], 2)) #选出text2末尾的sep_token_id
    # sep_index_second = sep_index.detach().cpu().numpy()[second_sep_index_list]#(batch, 2)

    # prepare ref_input_ids 
    # ref_input_ids = torch.LongTensor(batch_size,seq_len).zero_()
    # for i in range(batch_size):
    #     first_index = int(sep_index_first[i][1])
    #     second_index = int(sep_index_second[i][1])
    #     ref_input_ids[i][0] = cls_token_id
    #     ref_input_ids[i][first_index] = sep_token_id
    #     ref_input_ids[i][second_index] = sep_token_id
    
    attributions = lig.attribute(inputs = input_ids,
                                        baselines = ref_input_ids,
                                        target = batch_data["label"],
                                        n_steps = 20,
                                        additional_forward_args = (batch_data["attention_mask"], 
                                                                    batch_data["segemnet_ids"]))
#     attributions = torch.rand([batch_size, seq_len, 768]).to("cuda")
    
    # step 2: calculate the top k gradient vector
    # attributions.size: (batch, seq_len, ber_dim) , 对bert embedding归因
    # way 1: l2 norm
    encode_text = model.get_encode_text(batch_data).cpu()#(batch, seq, bert_dim)
    
    if not args.use_product:
        attributions_norm = torch.norm(attributions, dim = -1)#(batch, seq_len)
        top_k_indices = torch.topk(attributions_norm, k = args.bias_word_num).indices.cpu().numpy()#(batch, topk)
    # way 2: prodcut of bert and gradients 
    else:
        flat_encode_text = encode_text.unsqueeze(2)  #(batch,seq,1,bert_dim)
        attributions_vec = attributions.unsqueeze(-1) #(batch,seq_len,bert_dim,1)

        product_text = torch.matmul(flat_encode_text, attributions_vec).squeeze(-1).squeeze(-1) #(batch,seq)
        top_k_indices = torch.topk(product_text, k = args.bias_word_num).indices#(batch, topk)


    # step 3: flat top k vector
    batch_vec = []
    for i in range(top_k_indices.shape[0]):
        stack_vec = torch.cat([encode_text[i, top_k_indices[i][j]] for j in range(args.bias_word_num)], 
                            dim = -1).unsqueeze(0) #(top_k * bert_dim)
        batch_vec.append(stack_vec)
        
    batch_stack_vec = torch.cat([ele for ele in batch_vec], dim = 0)#(batch, top_k * bert_dim)
    
    return batch_stack_vec.to("cuda")



def int_grad_v2(args, model, batch_data, tokenizer):
    """
        this function is used for obataing the top k stack vector, including following steps:
            step 1: prepare for attribute function inputs, especially ref_input_ids
            step 2: calculate the top k gradient vector, 
                    this includes two ways: l2 norm or product of bert encode vector and gradient vector
            step 3: flat this top k vector and output it
        parameters: args
                    batch_data: inputs
                    model: orign_model
        return: flatten vector

    """
    def predict(input_ids, attention_mask, token_type_ids):
        return model.get_logits(input_ids, attention_mask, token_type_ids)
    # step 1: prepare for attribution inputs
    lig = LayerIntegratedGradients(predict, model.bert.embeddings)
    
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
    device = "cuda"
    input_ids = batch_data["input_ids"].to(device)#(batch, seq_len)
    batch_data["label"] = batch_data["label"].to(device)
    batch_data["attention_mask"] = batch_data["attention_mask"].to(device)
    batch_data["segemnet_ids"] = batch_data["segemnet_ids"].to(device)
    batch_size, seq_len = input_ids.size()
    ref_input_ids = input_ids.clone()
    ref_input_ids[(ref_input_ids != tokenizer.cls_token_id)*(ref_input_ids != tokenizer.sep_token_id)] = tokenizer.pad_token_id
    # sep_index = torch.nonzero(input_ids == sep_token_id)#(batch * 2, 2)
    # assert sep_index.size()[0] == input_ids.size()[0] * 2
    # first_sep_index_list = list(range(0, sep_index.size()[0], 2)) #选出text1末尾的sep_token_id
    # sep_index_first = sep_index.detach().cpu().numpy()[first_sep_index_list]#(batch,2) 2 代表(batch 中的第几个句子，middle_sep_index)
    # second_sep_index_list = list(range(1, sep_index.size()[0], 2)) #选出text2末尾的sep_token_id
    # sep_index_second = sep_index.detach().cpu().numpy()[second_sep_index_list]#(batch, 2)

    # prepare ref_input_ids 
    # ref_input_ids = torch.LongTensor(batch_size,seq_len).zero_()
    # for i in range(batch_size):
    #     first_index = int(sep_index_first[i][1])
    #     second_index = int(sep_index_second[i][1])
    #     ref_input_ids[i][0] = cls_token_id
    #     ref_input_ids[i][first_index] = sep_token_id
    #     ref_input_ids[i][second_index] = sep_token_id

    attributions = lig.attribute(inputs = input_ids,
                                        baselines = ref_input_ids,
                                        target = batch_data["label"],
                                        n_steps = 25,
                                        additional_forward_args = (batch_data["attention_mask"], 
                                                                    batch_data["segemnet_ids"]))
   
    
    # step 2: calculate the top k gradient vector
    # attributions.size: (batch, seq_len, ber_dim) , 对bert embedding归因
    # way 1: l2 norm
    encode_text = model.get_encode_text(batch_data)#(batch, seq, bert_dim)
    if not args.use_product:
        attributions_norm = torch.norm(attributions, dim = -1)#(batch, seq_len)
        top_k_indices = torch.topk(attributions_norm, k = args.bias_word_num).indices.cpu().numpy()#(batch, topk)
    # way 2: prodcut of bert and gradients 
    else:
        flat_encode_text = encode_text.unsqueeze(2)  #(batch,seq,1,bert_dim)
        attributions_vec = attributions.unsqueeze(-1) #(batch,seq_len,bert_dim,1)

        product_text = torch.matmul(flat_encode_text, attributions_vec).squeeze(-1).squeeze(-1) #(batch,seq)
        top_k_indices = torch.topk(product_text, k = args.bias_word_num).cpu().numpy()#(batch, topk)


    
    return top_k_indices