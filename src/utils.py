import numpy as np
import pytz
from datetime import datetime, timezone


def print_local_time():

    utc_dt = datetime.now(timezone.utc)
    PST = pytz.timezone('US/Pacific')
    print("Pacific time {}".format(utc_dt.astimezone(PST).isoformat()))

    return



def accuracy(pred,gt):

    pred = np.squeeze(pred[:,0])
    acc = np.sum(pred==gt)/len(gt)

    return acc


def mrr_score(pred,gt):

    mrr = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j]==gt[i]:
                mrr+=1/(j+1)
    mrr = mrr/len(gt)

    return mrr



def wu_p_score(pred, gt,path2root):

    pred = np.squeeze(pred[:,0])
    wu_p = 0
    for i in range(len(pred)):
        path_pred = path2root[pred[i]]
        path_gt = path2root[gt[i]]
        shared_nodes = set(path_pred)&set(path_gt)
        lca_depth = 1
        for node in shared_nodes:
            lca_depth = max(len(path2root[node]), lca_depth)
        wu_p+=2*lca_depth/(len(path_pred)+len(path_gt))
    
    wu_p = wu_p/len(gt)

    return wu_p
        

def metrics(pred, gt,path2root):

    acc = accuracy(pred,gt)
    mrr = mrr_score(pred,gt)
    wu_p = wu_p_score(pred, gt,path2root)


    return acc,mrr,wu_p