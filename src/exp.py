import os
import time
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from torch import optim
# import transformers
from transformers import BertTokenizer
from utils import *
from data import *
from model import BoxEmbed


class Experiments(object):

    def __init__(self,args):
        super(Experiments,self).__init__()
        
        self.args = args
        self.tokenizer = self.__load_tokenizer__()
        self.train_loader,self.train_set = load_data(self.args, self.tokenizer,"train")
        self.test_loader,self.test_set = load_data(self.args, self.tokenizer,"test")

        
        self.model = BoxEmbed(args,self.tokenizer)
        self.optimizer_pretrain, self.optimizer_projection = self._select_optimizer()
        self._set_device()
        self.exp_setting= str(self.args.pre_train)+"_"+str(self.args.dataset)+"_"+str(self.args.expID)+"_"+str(self.args.epochs)\
            +"_"+str(self.args.embed_size)+"_"+str(self.args.batch_size)+"_"+str(self.args.margin)+"_"+str(self.args.epsilon)\
                +"_"+str(self.args.size)+"_"+str(self.args.alpha)+"_"+str(self.args.beta)+"_"+str(self.args.gamma)+"_"+str(self.args.extra)
        setting={
            "pre_train":self.args.pre_train,
            "dataset":self.args.dataset,
            "expID":self.args.expID,
            "epochs":self.args.epochs,
            "embed_size":self.args.embed_size,
            "batch_size":self.args.batch_size,
            "margin":self.args.margin,
            "epsilon":self.args.epsilon,
            "size":self.args.size,
            "alpha":self.args.alpha,
            "beta":self.args.beta,
            "gamma":self.args.gamma,
            "extra":self.args.extra}
        print (setting)
        self.tosave_box={}
        self.tosave_pred={}


    def __load_tokenizer__(self):
        
        pre_trained_dic = {
            "bert": [BertTokenizer,"bert-base-uncased"]   
        }

        pre_train_tokenizer, checkpoint = pre_trained_dic[self.args.pre_train]
        tokenizer = pre_train_tokenizer.from_pretrained(checkpoint)

        return tokenizer



    def _select_optimizer(self):
        
        pre_train_parameters = [{"params": [p for n, p in self.model.named_parameters() if n.startswith("pre_train")],
                "weight_decay": 0.0},]
        projection_parameters = [{"params": [p for n, p in self.model.named_parameters() if n.startswith("projection")],
                "weight_decay": 0.0},]

        if self.args.optim=="adam":
            optimizer_pretrain = optim.Adam(pre_train_parameters, lr=self.args.lr)
            optimizer_projection = optim.Adam(projection_parameters, lr=self.args.lr_projection)
        elif self.args.optim=="adamw":
            optimizer_pretrain = optim.AdamW(pre_train_parameters,lr=self.args.lr, eps=self.args.eps)
            optimizer_projection = optim.AdamW(projection_parameters,lr=self.args.lr_projection, eps=self.args.eps)

        return optimizer_pretrain,optimizer_projection

    
    def _set_device(self):
        if self.args.cuda:
            self.model = self.model.cuda()




    def train_one_step(self,it,encode_parent, encode_child,encode_negative_parents):

        self.model.train()
        self.optimizer_pretrain.zero_grad()
        self.optimizer_projection.zero_grad()

        loss,loss_contain,loss_negative,regular_loss,loss_pos_prob,loss_neg_prob = self.model(encode_parent, encode_child,encode_negative_parents)
        loss.backward()
        self.optimizer_pretrain.step()
        self.optimizer_projection.step()

        return loss,loss_contain,loss_negative,regular_loss,loss_pos_prob,loss_neg_prob




    def train(self):
        
        time_tracker = []
        best_val_acc = best_epoch = test_acc = test_mrr = test_wu_p = 0
        for epoch in range(self.args.epochs):
            epoch_time = time.time()

            train_loss = []
            train_contain_loss = []
            train_negative_loss = []
            train_regular_loss = []
            train_pos_prob_loss = []
            train_neg_prob_loss = []


            for i, (encode_parent,encode_child,encode_negative_parents) in enumerate(self.train_loader):
                loss,loss_contain,loss_negative,regular_loss,loss_pos_prob,loss_neg_prob = self.train_one_step(it=i,encode_parent=encode_parent,encode_child=encode_child,encode_negative_parents=encode_negative_parents)
            
                train_loss.append(loss.item())
                train_contain_loss.append(loss_contain.item())
                train_negative_loss.append(loss_negative.item())
                train_regular_loss.append(regular_loss.item())
                train_pos_prob_loss.append(loss_pos_prob.item())
                train_neg_prob_loss.append(loss_neg_prob.item())
            

            train_loss = np.average(train_loss)
            train_contain_loss = np.average(train_contain_loss)
            train_negative_loss = np.average(train_negative_loss)
            train_regular_loss = np.average(train_regular_loss)
            train_pos_prob_loss = np.average(train_contain_loss)
            train_neg_prob_loss = np.average(train_neg_prob_loss)



            val_acc,val_mrr,val_wu_p = self.validation(flag="val")
            tmp_test_acc,tmp_test_mrr,tmp_test_wu_p = self.validation(flag="test")
            test_acc,test_mrr,test_wu_p = self.validation(flag="all")

            if val_acc > best_val_acc:
                
                best_epoch = epoch
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                test_mrr = tmp_test_mrr
                test_wu_p = tmp_test_wu_p

            
            
            
            time_tracker.append(time.time()-epoch_time)

            print('Epoch: {:04d}'.format(epoch + 1),
                 'Best epoch: {:04d}'.format(best_epoch + 1),
                ' train_loss:{:.05f}'.format(train_loss),
                'acc:{:.05f}'.format(test_acc),
                'mrr:{:.05f}'.format(test_mrr),
                'wu_p:{:.05f}'.format(test_wu_p),
                ' epoch_time:{:.01f}s'.format(time.time()-epoch_time),
                ' remain_time:{:.01f}s'.format(np.mean(time_tracker)*(self.args.epochs-(1+epoch))),
                )

        torch.save(self.model.state_dict(), os.path.join("../result",self.args.dataset,"model","exp_model_"+self.exp_setting+".checkpoint"))            

            


        




    def validation(self,flag):
        

        if flag == "val":
            encode_query = self.test_set.encode_val
            gt_label = self.test_set.val_gt
        elif flag == "test":
            encode_query = self.test_set.encode_test
            gt_label = self.test_set.test_gt
        elif flag == "all":
            encode_query = self.test_set.encode_query
            gt_label = self.test_set.test_gt_id           


        self.model.eval()
        score_list = []
        volumn_list = []
        contain_list = []
        with torch.no_grad():
            query_center,query_delta = self.model.projection_box(encode_query)
            num_query=len(query_center)
            for i in range(num_query):

                sorted_scores = []
                sorted_volumn = []
                hard_contain = []
                for j, (encode_candidate) in enumerate(self.test_loader):
                    candidate_center,candidate_delta = self.model.projection_box(encode_candidate)
                    num_candidate= len(candidate_center)

                    extend_center = [ query_center[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_delta = [ query_delta[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_center,extend_delta = torch.cat(extend_center,0),torch.cat(extend_delta,0)

                    score,volumn = self.model.condition_score(extend_center,extend_delta,candidate_center,candidate_delta)
                    is_contain = self.model.is_contain(extend_center,extend_delta,candidate_center,candidate_delta)

                    sorted_scores.append(score) 
                    sorted_volumn.append(volumn)
                    hard_contain.append(is_contain)
                sorted_scores = torch.cat(sorted_scores)
                sorted_volumn = torch.cat(sorted_volumn)
                hard_contain = torch.cat(hard_contain)

                score_list.append(sorted_scores.unsqueeze(dim=0))
                volumn_list.append(sorted_volumn.unsqueeze(dim=0))
                contain_list.append(hard_contain.unsqueeze(dim=0))
            
            pred_scores = torch.cat(score_list,0)
            pred_volumn = torch.cat(volumn_list,0)
            pred_contain = torch.cat(contain_list,0)
            pred_scores,pred_volumn = pred_scores.detach().cpu().numpy(), pred_volumn.detach().cpu().numpy()
            pred_contain = pred_contain.detach().cpu().numpy()
            ind = np.lexsort((pred_volumn,pred_scores*(-1))) # Sort by pred_scores, then by pred_volumn

            x,y = pred_scores.shape
            pred = np.array([[i for i in range(y)] for _ in range(x)])
            
            for i in range(len(pred)):
                pred[i]=np.array(list(self.train_set.train_concept_set))[pred[i][ind[i]]]

            
            acc,mrr,wu_p = metrics(pred,gt_label,self.test_set.path2root)


        return acc,mrr,wu_p




    def predict(self):
        
        print ("Prediction starting.....")
        self.model.load_state_dict(torch.load(os.path.join("../result",self.args.dataset,"model","exp_model_"+self.exp_setting+".checkpoint")))
        self.model.eval()
        score_list = []
        volumn_list = []
        contain_list = []
        with torch.no_grad():
            query_center,query_delta = self.model.projection_box(self.test_set.encode_query)
            num_query=len(query_center)
            for i in range(num_query):

                sorted_scores = []
                sorted_volumn = []
                hard_contain = []
                for j, (encode_candidate) in enumerate(self.test_loader):
                    candidate_center,candidate_delta = self.model.projection_box(encode_candidate)
                    num_candidate= len(candidate_center)

                    extend_center = [ query_center[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_delta = [ query_delta[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                    extend_center,extend_delta = torch.cat(extend_center,0),torch.cat(extend_delta,0)

                    score,volumn = self.model.condition_score(extend_center,extend_delta,candidate_center,candidate_delta)
                    is_contain = self.model.is_contain(extend_center,extend_delta,candidate_center,candidate_delta)

                    sorted_scores.append(score) 
                    sorted_volumn.append(volumn)
                    hard_contain.append(is_contain)
                sorted_scores = torch.cat(sorted_scores)
                sorted_volumn = torch.cat(sorted_volumn)
                hard_contain = torch.cat(hard_contain)

                score_list.append(sorted_scores.unsqueeze(dim=0))
                volumn_list.append(sorted_volumn.unsqueeze(dim=0))
                contain_list.append(hard_contain.unsqueeze(dim=0))
            
            pred_scores = torch.cat(score_list,0)
            pred_volumn = torch.cat(volumn_list,0)
            pred_contain = torch.cat(contain_list,0)
            pred_scores,pred_volumn = pred_scores.detach().cpu().numpy(), pred_volumn.detach().cpu().numpy()
            pred_contain = pred_contain.detach().cpu().numpy()
            ind = np.lexsort((pred_volumn,pred_scores*(-1))) # Sort by pred_scores, then by pred_volumn

            x,y = pred_scores.shape
            pred = np.array([[i for i in range(y)] for _ in range(x)])
            print (pred.shape)
            
            for i in range(len(pred)):
                pred[i]=np.array(list(self.train_set.train_concept_set))[pred[i][ind[i]]]

            acc,mrr,wu_p = metrics(pred,self.test_set.test_gt_id,self.test_set.path2root)
        
        print('score: acc: {:.05f}'.format(acc),
                'mrr:{:.05f}'.format(mrr),
                'wu_p:{:.05f}'.format(wu_p),)


        self.tosave_pred["metric"] = (acc,mrr,wu_p)
        self.tosave_pred["pred"] = pred
        self.tosave_pred["pred2"] = 0
        self.tosave_pred["pred_scores"] = pred_scores
        self.tosave_pred["pred_volumn"] = pred_volumn
        self.tosave_pred["pred_contain"] = pred_contain
        self.tosave_pred["gt"] = self.test_set.test_gt_id
        self.tosave_pred["path2root"]=self.test_set.path2root


        return 



    def save_prediction(self):
        
        self.model.eval()
        encode_dic = self.train_set.encode_all
        input_ids,token_type_ids,attention_mask = encode_dic["input_ids"],encode_dic["token_type_ids"],encode_dic["attention_mask"]
        length = self.args.batch_size
        l = 0
        r = length
        center_list = []
        delta_list = []
        with torch.no_grad():
            while l < (len(input_ids)):
                r = min (r,len(input_ids))
                encode = {
                    "input_ids":input_ids[l:r],
                    "token_type_ids":token_type_ids[l:r],
                    "attention_mask":attention_mask[l:r]
                }
                center,delta = self.model.projection_box(encode)
                center = center.detach().cpu().numpy()
                delta = delta.detach().cpu().numpy()
                center_list.append(center)
                delta_list.append(delta)
                l = r
                r+=length
        center = np.concatenate(center_list)
        delta = np.concatenate(delta_list)
                

        self.tosave_box["left"] = center-delta
        self.tosave_box["right"] = center+delta
        self.tosave_box["center"] = center
        self.tosave_box["delta"] = delta

        with open(os.path.join("../result",self.args.dataset,"box","exp_box_"+self.exp_setting+".pkl"),"wb") as f:
            pkl.dump(self.tosave_box,f)

        with open(os.path.join("../result",self.args.dataset,"prediction","exp_pred_"+self.exp_setting+".pkl"),"wb") as f:
            pkl.dump(self.tosave_pred,f)
        

        print ("================================Save results done!================================")

