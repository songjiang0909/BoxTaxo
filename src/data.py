import os
from re import L
import time
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Data_TRAIN(Dataset):
    
    def __init__(self,args,tokenizer):
        
        super(Data_TRAIN, self).__init__()

        self.args = args
        self.dataset = args.dataset
        print ("Dataset: {}".format(self.dataset))
        self.data = self.__load_data__(self.dataset)
        self.tokenizer = tokenizer

        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]
        self.train_concept_set = self.data["train_concept_set"]
        self.train_parent_list = self.data["train_parent_list"]
        self.train_child_list = self.data["train_child_list"]
        self.train_negative_parent_dict = self.data["train_negative_parent_dict"]
        self.train_sibling_dict = self.data["train_sibling_dict"]
        self.child_parent_pair = self.data["child_parent_pair"]
        self.child_neg_parent_pair = self.data["child_neg_parent_pair"]
        self.child_sibling_pair = self.data["child_sibling_pair"]


        self.train_child_parent_negative_parent_triple = self.data["train_child_parent_negative_parent_triple"]


        if self.args.model == "vec":
            self.train_child_parent_negative_parent_triple = self.negative_sampling()
        print ("Training samples: {}".format(len(self.train_child_parent_negative_parent_triple)))



        self.encode_all = self.generate_all_token_ids(self.tokenizer)



    def __load_data__(self,dataset):
        
        
        with open(os.path.join("../data/",dataset,"processed","taxonomy_data_"+str(self.args.expID)+"_.pkl"),"rb") as f:
            data = pkl.load(f)
        
        return data




    def align_data(self):
        
        np.random.shuffle(self.child_parent_pair)
        np.random.shuffle(self.child_neg_parent_pair)
        np.random.shuffle(self.child_sibling_pair)

        child_parent_pair = (self.child_parent_pair*self.max_len)[:self.max_len]
        child_neg_parent_pair = (self.child_neg_parent_pair*self.max_len)[:self.max_len]
        child_sibling_pair = (self.child_sibling_pair*self.max_len)[:self.max_len]

        return np.concatenate((np.array(child_parent_pair),np.array(child_neg_parent_pair),np.array(child_sibling_pair)),axis=1), self.max_len


    def align_data_sample(self):
        
        np.random.shuffle(self.child_parent_pair)
        np.random.shuffle(self.child_neg_parent_pair)
        np.random.shuffle(self.child_sibling_pair)
        
        sample_num = self.min_len*self.args.sample
        child_parent_pair = (self.child_parent_pair*self.max_len)[:sample_num]
        child_neg_parent_pair = (self.child_neg_parent_pair*self.max_len)[:sample_num]
        child_sibling_pair = (self.child_sibling_pair*self.max_len)[:sample_num]

        return np.concatenate((np.array(child_parent_pair),np.array(child_neg_parent_pair),np.array(child_sibling_pair)),axis=1), sample_num





    def negative_sampling(self):
        childs_query = []
        parent_as_positive = []
        negative_parent_list = []
        child_sibling = []
        sample_num = self.args.sample
        for i in range(len(self.train_child_list)):
            cid = self.train_child_list[i]
            pid = self.train_parent_list[i]
            negative_set = list(self.train_negative_parent_dict[cid])
            sibling_set = list(self.train_sibling_dict[cid])

            if np.array(negative_set).shape[0]==0:
                continue
            np.random.shuffle(negative_set)
            negative_set = negative_set[:sample_num]

            if np.array(sibling_set).shape[0]==0:
                continue
            np.random.shuffle(sibling_set)
            sibling_set = sibling_set[:sample_num]

            max_l = max(len(negative_set),len(sibling_set))
            negative_set = (negative_set*max_l)[:max_l]
            sibling_set = (sibling_set*max_l)[:max_l]
            

            for k in range(max_l):
                childs_query.append(cid)
                parent_as_positive.append(pid)
                negative_parent_list.append(negative_set[k])
                child_sibling.append(sibling_set[k])



        child_parent_negative_parent_triple = np.stack((childs_query,parent_as_positive,negative_parent_list),axis=0).T
        child_parent_negative_parent_triple = child_parent_negative_parent_triple.tolist()

        return child_parent_negative_parent_triple




    def generate_all_token_ids(self,tokenizer):

        all_nodes_context = [self.id_context[cid] for cid in self.concept_set]
        encode_all = tokenizer(all_nodes_context, padding=True,return_tensors='pt')
        
        if self.args.cuda:
            a_input_ids = encode_all['input_ids'].cuda()
            a_token_type_ids = encode_all['token_type_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {'input_ids' : a_input_ids, 
                        'token_type_ids' : a_token_type_ids, 
                        'attention_mask' : a_attention_mask} 
        return encode_all




    def index_token_ids(self,encode_dic,index):

        input_ids,token_type_ids,attention_mask = encode_dic["input_ids"],encode_dic["token_type_ids"],encode_dic["attention_mask"]
        
        res_dic = {'input_ids' : input_ids[index], 
                        'token_type_ids' : token_type_ids[index], 
                        'attention_mask' : attention_mask[index]}


        return res_dic




    def generate_parent_child_token_ids(self,index):

        child_id,parent_id,negative_parent_id = self.train_child_parent_negative_parent_triple[index]
        encode_child = self.index_token_ids(self.encode_all,child_id)
        encode_parent = self.index_token_ids(self.encode_all,parent_id)
        encode_negative_parents = self.index_token_ids(self.encode_all,negative_parent_id)

        return encode_parent, encode_child,encode_negative_parents


    def __getitem__(self, index):


        encode_parent, encode_child,encode_negative_parents = self.generate_parent_child_token_ids(index)

        return encode_parent, encode_child,encode_negative_parents



    def __len__(self):
        
        return len(self.train_child_parent_negative_parent_triple)






class Data_TEST(Dataset):
    
    def __init__(self,args,tokenizer):
        
        super(Data_TEST, self).__init__()

        self.args = args
        self.dataset = args.dataset
        print ("Dataset: {}".format(self.dataset))
        self.data = self.__load_data__(self.dataset)
        self.tokenizer = tokenizer

        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]

        self.train_concept_set = list(self.data["train_concept_set"])
        self.path2root = self.data["path2root"]
        self.test_concepts_id = self.data["test_concepts_id"]
        self.test_gt_id = self.data["test_gt_id"]

        self.val_concept,self.val_gt, self.test_concept, self.test_gt = self.val_test_split()

        self.encode_all = self.generate_all_token_ids(self.tokenizer)
        
        self.encode_val = self.generate_test_token_ids(self.tokenizer,self.val_concept)
        self.encode_test = self.generate_test_token_ids(self.tokenizer,self.test_concept)

        self.encode_query = self.generate_test_token_ids(self.tokenizer,self.test_concepts_id)



    def __load_data__(self,dataset):
        
        with open(os.path.join("../data/",dataset,"processed","taxonomy_data_"+str(self.args.expID)+"_.pkl"),"rb") as f:
            data = pkl.load(f)
        
        return data



    def val_test_split(self):

        tmp = list(zip(self.test_concepts_id, self.test_gt_id))
        np.random.shuffle(tmp)
        num = int(len(self.test_concepts_id)*0.3333)
        shuffled_concept, shuffled_gt = zip(*tmp)
        val_concept, val_gt = shuffled_concept[:num], shuffled_gt[:num]
        test_concept, test_gt = shuffled_concept[num:], shuffled_gt[num:]

        print ("validation num: {}".format(len(val_concept)))
        print ("test num: {}".format(len(test_concept)))

        return val_concept, val_gt, test_concept, test_gt


    def generate_all_token_ids(self,tokenizer):

        all_nodes_context = [self.id_context[cid] for cid in self.concept_set]
        encode_all = tokenizer(all_nodes_context, padding=True,return_tensors='pt')
        if self.args.cuda:
            a_input_ids = encode_all['input_ids'].cuda()
            a_token_type_ids = encode_all['token_type_ids'].cuda()
            a_attention_mask = encode_all['attention_mask'].cuda()

            encode_all = {'input_ids' : a_input_ids, 
                        'token_type_ids' : a_token_type_ids, 
                        'attention_mask' : a_attention_mask} 

        return encode_all



    def generate_test_token_ids(self,tokenizer, test_concepts_id):

        test_nodes_context = [self.id_context[cid] for cid in test_concepts_id]
        encode_test = tokenizer(test_nodes_context, padding=True,return_tensors='pt')
        if self.args.cuda:
            t_input_ids = encode_test['input_ids'].cuda()
            t_token_type_ids = encode_test['token_type_ids'].cuda()
            t_attention_mask = encode_test['attention_mask'].cuda()

            encode_test = {'input_ids' : t_input_ids, 
                        'token_type_ids' : t_token_type_ids, 
                        'attention_mask' : t_attention_mask} 

        return encode_test



    def index_token_ids(self,encode_dic,index):

        input_ids,token_type_ids,attention_mask = encode_dic["input_ids"],encode_dic["token_type_ids"],encode_dic["attention_mask"]
        
        res_dic = {'input_ids' : input_ids[index], 
                        'token_type_ids' : token_type_ids[index], 
                        'attention_mask' : attention_mask[index]}


        return res_dic



    def __getitem__(self, index):

        candidate_ids = self.train_concept_set[index]

        encode_candidate = self.index_token_ids(self.encode_all,candidate_ids)

        return encode_candidate



    def __len__(self):

        return len(self.train_concept_set)






def load_data(args, tokenizer,flag):

    if flag in set(['test','val']) :
        shuffle_flag = False; drop_last = False; batch_size = args.batch_size; 
        data_set = Data_TEST(args,tokenizer)
    else:
        shuffle_flag = True; drop_last = False; batch_size = args.batch_size; 
        data_set = Data_TRAIN(args,tokenizer)
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)

    return data_loader,data_set