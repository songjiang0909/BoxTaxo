import os
import time
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    
    def __init__(self,args):
        
        super(Data, self).__init__()

        self.dataset = args.dataset
        print ("Dataset: {}".format(self.dataset))
        self.data = self.__load_data__(self.dataset)
        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]

        self.train_concept_set = self.data["train_concept_set"]
        self.train_taxo_dict = self.data["train_taxo_dict"]
        self.train_negative_parent_dict = self.data["train_negative_parent_dict"]
        self.train_parent_list = self.data["train_parent_list"]
        self.train_child_list = self.data["train_child_list"]
        self.train_child_parent_negative_parent_triple = self.data["train_child_parent_negative_parent_triple"]
        self.path2root = self.data["path2root"]


        self.test_concepts_id = self.data["test_concepts_id"]
        self.test_gt_id = self.data["test_gt_id"]


    def __load_data__(self,dataset):
        
        with open(os.path.join("../data/",dataset,"processed","taxonomy_data.pkl"),"rb") as f:
            data = pkl.load(f)
        
        return data
        

    def __getitem__(self, index):


        
        return self.data

    def __len__(self):
        return 1






def load_data(args, flag):

    if flag in set(['test','val']) :
        shuffle_flag = False; drop_last = False; batch_size = args.batch_size; 
    else:
        shuffle_flag = True; drop_last = False; batch_size = args.batch_size; 
    
    data_set = Data(args=args)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)

    return data_set, data_loader