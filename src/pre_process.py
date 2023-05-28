
import os
import json
import pickle as pkl
import numpy as np
import collections
from random import shuffle
import time


def preprocess(args,outID=True):

    dataset = args.dataset
    with open(os.path.join("../data/"+str(dataset)+"/"+str(dataset)+"_raw_en.taxo")) as f:
        taxonomy = f.readlines()        
    concept_set = set([])
    all_taxo_dict = collections.defaultdict(list)
    for pair in taxonomy:
        _, child, parent = pair.split("\n")[0].split("\t")
        concept_set.add(parent)
        concept_set.add(child)
        
    concept_index = range(len(concept_set))
    concepts = sorted(list(concept_set))
    concept_id = dict(zip(concepts,concept_index))
    id_concept = dict(zip(concept_index,concepts))
    if outID:
        concept_set = set([concept_id[con] for con in list(concept_set)])
        for pair in taxonomy:
            _, child, parent = pair.split("\n")[0].split("\t")
            all_taxo_dict[concept_id[parent]].append(concept_id[child])

    train_concept_set = set([])
    print ("loading training data")
    with open("../data/"+str(dataset)+"/"+str(dataset)+"_train.taxo") as f:
        train_taxonomy = f.readlines()        

    parent_list = []
    child_list = []
    chd2par_dict = collections.defaultdict(set)
    taxo_dict = collections.defaultdict(list)
    root_id = dataset


    if outID:
        root_id = concept_id[root_id]
    for pair in train_taxonomy:
        parent, child = pair.split("\n")[0].split("\t")
        if outID:
            parent,child = concept_id[parent],concept_id[child]
        parent_list.append(parent)
        child_list.append(child)
        train_concept_set.add(parent)
        train_concept_set.add(child)
        chd2par_dict[child].add(parent) 
        taxo_dict[parent].append(child)

    cnt_dic = collections.defaultdict(float)
    for _,parent in chd2par_dict.items():
        cnt_dic[len(parent)]+=1


    sibling_dict = collections.defaultdict(set)
    for parent,child in taxo_dict.items():
        for node in child:
            sibling_dict[node] = sibling_dict[node] | (set(taxo_dict[parent])-set([node])) 
    observe_nodes = train_concept_set - (set([root_id]) | set(taxo_dict[root_id]))
    
    sib_pair = []
    for k,c in sibling_dict.items():
        for l in c:
            sib_pair.append([k,l])


    cousin_dict = collections.defaultdict(set)
    for node in observe_nodes:
        pars = chd2par_dict[node]
        for par in pars:
            cousin_dict[node] =cousin_dict[node] | (sibling_dict[par]-pars)
            uncles = cousin_dict[node] | (sibling_dict[par]-pars)

        for uncle in uncles:
            cousin_dict[node] =cousin_dict[node] | set(taxo_dict[uncle])    
            cousin_dict[node] = cousin_dict[node] - sibling_dict[node]


    relative_triple = []
    for node in observe_nodes:
        sibling = list(sibling_dict[node])
        cousin = list(cousin_dict[node])
        for s in sibling:
            for c in cousin:
                relative_triple.append([node,s,c])


    negative_parent_dict = collections.defaultdict(set)
    for cid,_ in id_concept.items():
        negative_parent_dict[cid] = sibling_dict[cid] | cousin_dict[cid]

    child_for_negative = []
    parent_as_positive = []
    negative_parent_list = []
    for i in range(len(child_list)):
        cid = child_list[i]
        pid = parent_list[i]


        negative_set = negative_parent_dict[cid]
        for negative_parent in list(negative_set):
            child_for_negative.append(cid)
            parent_as_positive.append(pid)
            negative_parent_list.append(negative_parent)
    
    child_parent_negative_parent_triple = np.stack((child_for_negative,parent_as_positive,negative_parent_list),axis=0).T
    child_parent_negative_parent_triple = child_parent_negative_parent_triple.tolist()  


    child_parent_pair = []
    for i in range(len(child_list)):
        child_parent_pair.append([child_list[i],parent_list[i]])

    
    child_neg_parent_pair = []
    for i in range(len(child_list)):
        cid = child_list[i]
        negative_set = negative_parent_dict[cid]
        for negative_parent in list(negative_set):
            child_neg_parent_pair.append([cid,negative_parent])

    child_sibling_pair = []
    for i in range(len(child_list)):
        cid = child_list[i]
        sib_set = sibling_dict[cid]
        for sib in list(sib_set):
            child_sibling_pair.append([cid,sib])


        

    with open("../data/"+str(dataset)+"/dic.json") as f:
        def_dic = json.load(f)
    
    id_context = collections.defaultdict(str)
    for cid,concept in id_concept.items():
        context = concept + ": " + def_dic[concept][0]
        id_context[cid] = context

    print ("loading testing data")
    with open("../data/"+str(dataset)+"/"+str(dataset)+"_eval.terms") as f:
        test_terms = f.readlines()
    with open("../data/"+str(dataset)+"/"+str(dataset)+"_eval.gt") as f:
        test_gt = f.readlines()        

    test_concepts_id = []
    test_gt_id = []
    for term in test_terms:
        term_id = concept_id[term.split("\n")[0]]
        test_concepts_id.append(term_id)
    for term in test_gt:
        term_id = concept_id[term.split("\n")[0]]
        test_gt_id.append(term_id)


    # valid and test
    tmp = list(zip(test_concepts_id, test_gt_id))
    np.random.shuffle(tmp)
    num = int(len(test_concepts_id)*0.5)
    shuffled_concept, shuffled_gt = zip(*tmp)
    val_concept, val_gt = shuffled_concept[:num], shuffled_gt[:num]
    test_concept, test_gt = shuffled_concept[num:], shuffled_gt[num:]

    path2root =  collections.defaultdict(list)
    for node in train_concept_set:
        path2root[node].append(node)
        if node == root_id:
            continue
        parent_node = list(chd2par_dict[node])[0]
        path2root[node].append(parent_node)
        while parent_node!=root_id:
            parent_node = list(chd2par_dict[parent_node])[0]
            path2root[node].append(parent_node)



    return concept_set,concept_id,id_concept,id_context,train_concept_set,taxo_dict,negative_parent_dict,child_parent_negative_parent_triple,\
        parent_list,child_list,negative_parent_list,sibling_dict,cousin_dict,relative_triple,test_concepts_id,test_gt_id,all_taxo_dict,path2root,sib_pair,\
            child_parent_pair,child_neg_parent_pair,child_sibling_pair,val_concept, val_gt,test_concept, test_gt



def create_data(args):


    concept_set,concept_id,id_concept,id_context,train_concept_set,train_taxo_dict,negative_parent_dict,train_child_parent_negative_parent_triple,train_parent_list,\
    train_child_list,train_negative_parent_list,train_sibling_dict,train_cousin_dict,train_relative_triple,test_concepts_id,test_gt_id,\
        all_taxo_dict,path2root,sib_pair,child_parent_pair,child_neg_parent_pair,child_sibling_pair,val_concept, val_gt,test_concept, test_gt = preprocess(args)
    
    print ("Waitfing for preprocess data....")
    time.sleep(3)
    print ("Done!")
    save_data = {
    "concept_set":concept_set,
    "concept2id":concept_id,
    "id2concept":id_concept,
    "id2context":id_context,
    "all_taxo_dict":all_taxo_dict,
    "train_concept_set":train_concept_set,
    "train_taxo_dict":train_taxo_dict,
    "train_negative_parent_dict":negative_parent_dict,
    "train_child_parent_negative_parent_triple":train_child_parent_negative_parent_triple,
    "train_parent_list":train_parent_list,
    "train_child_list":train_child_list,
    "train_negative_parent_list":train_negative_parent_list,
    "train_sibling_dict":train_sibling_dict,
    "train_cousin_dict":train_cousin_dict,
    "train_relative_triple":train_relative_triple,
    "test_concepts_id":test_concepts_id,
    "test_gt_id":test_gt_id,
    "path2root":path2root,
    "sib_pair":sib_pair,
    "child_parent_pair":child_parent_pair,
    "child_neg_parent_pair":child_neg_parent_pair,
    "child_sibling_pair":child_sibling_pair,
    "val_concept":val_concept, 
    "val_gt":val_gt,
    "test_concept":test_concept, 
    "test_gt":test_gt}

    with open("../data/"+str(args.dataset)+"/processed/taxonomy_data_"+str(args.expID)+"_.pkl","wb") as f:
        pkl.dump(save_data,f)

    print ("Waitfing for saving processed data....")
    time.sleep(3)
    print ("Done!")
    print (f"From processed data, there are :{len(train_child_parent_negative_parent_triple)} training instances")
    print (f"From processed data, there are :{len(test_gt_id)} test instances")

    

    
    