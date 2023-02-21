import random
import os
import pickle as pkl
import torch
import torch.nn as nn
from utils import *
from layers import MLP
from transformers import BertTokenizer, BertModel



class BoxEmbed(nn.Module):

    def __init__(self,args,tokenizer):

        super(BoxEmbed, self).__init__()

        self.args = args
        self.data = self.__load_data__(self.args.dataset)
        self.FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]

        self.train_concept_set = list(self.data["train_concept_set"])
        self.train_taxo_dict = self.data["train_taxo_dict"]
        self.train_child_parent_negative_parent_triple = self.data["train_child_parent_negative_parent_triple"]
        self.path2root = self.data["path2root"]
        self.test_concepts_id = self.data["test_concepts_id"]
        self.test_gt_id = self.data["test_gt_id"]


        self.pre_train_model = self.__load_pre_trained__()

        self.projection_center = MLP(input_dim=768,hidden=self.args.hidden,output_dim=self.args.embed_size)
        self.projection_delta = MLP(input_dim=768,hidden=self.args.hidden,output_dim=self.args.embed_size) 

        
        self.dropout = nn.Dropout(self.args.dropout)

        
        self.par_chd_left_loss = nn.MSELoss()
        self.par_chd_right_loss = nn.MSELoss()
        self.par_chd_negative_loss = nn.MSELoss()
        self.box_size_loss = nn.MSELoss()
        self.positive_prob_loss = nn.MSELoss()
        self.negative_prob_loss = nn.MSELoss()



    def __load_data__(self,dataset):
        
        with open(os.path.join("../data/",dataset,"processed","taxonomy_data_"+str(self.args.expID)+"_.pkl"),"rb") as f:
            data = pkl.load(f)
        
        return data



    def __load_pre_trained__(self):
        
        pre_trained_dic = {
            "bert": [BertModel,"bert-base-uncased"]
        }

        pre_train_model, checkpoint = pre_trained_dic[self.args.pre_train]
        model = pre_train_model.from_pretrained(checkpoint)

        return model

    
        

    def parent_child_contain_loss(self,parent_center,parent_delta,child_center,child_delta):

        parent_left = parent_center-parent_delta
        parent_right = parent_center+parent_delta

        child_left = child_center-child_delta
        child_right = child_center+child_delta


        diff_left = child_left-parent_left
        zeros = torch.zeros_like(diff_left)
        ones = torch.ones_like(diff_left)
        margins = torch.ones_like(diff_left)*self.args.margin
        left_mask = torch.where(diff_left < self.args.margin, ones, zeros)
        left_loss = self.par_chd_left_loss(torch.mul(diff_left,left_mask),torch.mul(margins,left_mask))


        diff_right = parent_right-child_right
        zeros = torch.zeros_like(diff_right)
        ones = torch.ones_like(diff_right)
        margins = torch.ones_like(diff_right)*self.args.margin
        right_mask = torch.where(diff_right < self.args.margin, ones, zeros)
        right_loss = self.par_chd_right_loss(torch.mul(diff_right,right_mask),torch.mul(margins,right_mask))

        return (left_loss+right_loss)/2



    def parent_child_contain_loss_prob(self,parent_center,parent_delta,child_center,child_delta):
        
        score,_ = self.condition_score(child_center,child_delta,parent_center,parent_delta)
        ones = torch.ones_like(score)
        loss = self.positive_prob_loss(score,ones)

        return loss



    def box_intersection(self,center1,delta1,center2,delta2):

        left1 = center1-delta1
        right1 = center1+delta1
        left2 = center2-delta2
        right2 = center2+delta2
        inter_left = torch.max(left1,left2)
        inter_right = torch.min(right1,right2)
        

        return inter_left,inter_right


    def negative_contain_loss(self,child_center,child_delta,neg_parent_center, neg_parent_delta):

        inter_left,inter_right = self.box_intersection(child_center,child_delta,neg_parent_center, neg_parent_delta)
        
        inter_delta = (inter_right-inter_left)/2
        zeros = torch.zeros_like(inter_delta)
        ones = torch.ones_like(inter_delta)
        epsilon = torch.ones_like(inter_delta)*self.args.epsilon
        inter_mask = torch.where(inter_delta > self.args.epsilon, ones, zeros)
        inter_loss = self.par_chd_negative_loss(torch.mul(inter_delta,inter_mask),torch.mul(epsilon,inter_mask))

        return inter_loss


    def negative_contain_loss_prob(self,child_center,child_delta,neg_parent_center, neg_parent_delta):

        score,_ = self.condition_score(child_center,child_delta,neg_parent_center,neg_parent_delta)
        zeros = torch.zeros_like(score)
        loss = self.negative_prob_loss(score,zeros)

        return loss



    def sibling_distance_loss(self,child_center, sibling_center):
        
        if self.args.sim_fun == "cos":
            distance = self.box_center_distance_cos(child_center, sibling_center)
        elif self.args.sim_fun == "eu":
            distance = self.box_center_distance(child_center, sibling_center)
        
        zeros = torch.zeros_like(distance)
        ones = torch.ones_like(distance)
        margin = torch.ones_like(distance)*self.args.dis_margin
        if self.args.sim_fun == "cos":
            dis_mask = torch.where(distance != self.args.dis_margin, ones, zeros)
        elif self.args.sim_fun == "eu":
            dis_mask = torch.where(distance < self.args.dis_margin, ones, zeros)
        distance_loss = self.dis_loss(torch.mul(distance,dis_mask),torch.mul(margin,dis_mask))

        return distance_loss



    def box_center_distance(self,center1, center2):
        radius = center1-center2
        return torch.linalg.norm(radius,2,-1)


    def box_center_distance_cos(self,center1, center2):

        cos = nn.CosineSimilarity()
        return cos(center1,center2)



    def projection_box(self,encode_inputs):
        cls = self.pre_train_model(**encode_inputs)
        cls = self.dropout(cls[0][:, 0, :])
        center = self.projection_center(cls)
        delta = torch.exp(self.projection_delta(cls)).clamp_min(1e-38)

        return center,delta



    def box_volumn(self,delta):

        flag = torch.sum(delta<=0,1)
        product = torch.prod(delta,1)
        zeros = torch.zeros_like(product)
        ones = torch.ones_like(product)
        mask = torch.where(flag==0, ones, zeros)
        volumn = torch.mul(product,mask)


        return volumn



    def box_regularization(self,delta):


        zeros = torch.zeros_like(delta)
        ones = torch.ones_like(delta)
        mini_size = torch.ones_like(delta)*self.args.size
        inter_mask = torch.where(delta < self.args.size, ones, zeros)
        regular_loss = self.box_size_loss(torch.mul(delta,inter_mask),torch.mul(mini_size,inter_mask))

        return regular_loss


    def condition_score(self, child_center,child_delta,parent_center,parent_delta):

        inter_left,inter_right = self.box_intersection(child_center,child_delta,parent_center,parent_delta)
        inter_delta = (inter_right-inter_left)/2
        flag = (inter_delta<=0)
        zeros = torch.zeros_like(flag)
        ones = torch.ones_like(flag)
        mask = torch.where(flag==False, ones, zeros)
        masked_inter_delta = torch.mul(inter_delta,mask)
        score_pre = torch.div(masked_inter_delta,child_delta)
        score = torch.prod(score_pre,1)
        

        parent_volumn = self.box_volumn(parent_delta)

        return score.squeeze(),parent_volumn.squeeze()


    def is_contain(self, child_center,child_delta,parent_center,parent_delta):

        child_left = child_center-child_delta
        child_right = child_center+child_delta
        parent_left = parent_center-parent_delta
        parent_right = parent_center+parent_delta

        flag = (torch.sum(child_left>=parent_left,1)+torch.sum(child_right<=parent_right,1))==child_left.shape[1]*2
        zeros = torch.zeros_like(flag)
        ones = torch.ones_like(flag)
        mask = torch.where(flag, ones, zeros)

        return mask.squeeze()



    def forward(self,encode_parent=None,encode_child=None,encode_negative_parents=None,flag="train"):

        if flag == "train":
            regular_loss = 0

            parent_center,parent_delta = self.projection_box(encode_parent)
            child_center, child_delta = self.projection_box(encode_child)
            parent_child_contain_loss = self.parent_child_contain_loss(parent_center,parent_delta,child_center,child_delta)
            parent_child_contain_loss_prob = self.parent_child_contain_loss_prob(parent_center,parent_delta,child_center,child_delta)

            neg_parent_center, neg_parent_delta = self.projection_box(encode_negative_parents)
            child_parent_negative_loss = self.negative_contain_loss(child_center,child_delta,neg_parent_center, neg_parent_delta)
            child_parent_negative_loss_prob = self.negative_contain_loss_prob(child_center,child_delta,neg_parent_center, neg_parent_delta)


            regular_loss += self.box_regularization(parent_delta)
            regular_loss += self.box_regularization(child_center)
            regular_loss += self.box_regularization(neg_parent_delta)


            
            loss_contain = self.args.alpha*parent_child_contain_loss 
            loss_negative = self.args.alpha*child_parent_negative_loss
            regular_loss = self.args.gamma*regular_loss
            loss_pos_prob = self.args.extra*parent_child_contain_loss_prob
            loss_neg_prob = self.args.extra*child_parent_negative_loss_prob
            


            loss = loss_contain+loss_negative+regular_loss
            loss+=loss_pos_prob
            loss+=loss_neg_prob

        

        return loss,loss_contain,loss_negative,regular_loss,loss_pos_prob,loss_neg_prob





        


