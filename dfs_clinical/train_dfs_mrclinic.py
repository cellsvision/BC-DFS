import pandas as pd
from tqdm import tqdm
import argparse
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from sksurv.datasets import load_whas500
from sksurv.linear_model import CoxPHSurvivalAnalysis,CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.utils import save_model,load_model
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score,roc_curve

from monai.networks.nets import FullyConnectedNet
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

import CoxPHLoss

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cv2

from qlib.contrib.model import pytorch_tabnet


import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', help='train csv path', required=False, type=str,default='data/sample_dfsmrclinic.csv')
    parser.add_argument('--val_list', help='val csv path', required=False, type=str,default='data/sample_dfsmrclinic.csv')
    args = parser.parse_args()
    return args

args = parse_args()
val_list = args.val_list
train_list = args.train_list
train_df = pd.read_csv(train_list,dtype={'ID':str})
test_df = pd.read_csv(val_list,dtype={'ID':str})

all_feas = ['risk_score','n_tumors','HER2_status','subtype','age','ER_expression', 'PR_expression','Ki67_expression','T_stage','N_stage','pTNM'] 

n_steps = 3
n_d = n_a = 14
relax = 1.25

ckpt_root = './ckpts'

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()*(-1)
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)



def get_timeDependent_auc(y_T, y_E, pred_score, times=[12,24,36,48],exclude_range=0):
    time_dep_auc_all = {}
    for timedep in times:
        tmp_e = []
        tmp_pred = []
        tmp_t = []
        for i in range(len(y_T)):
            if y_T[i]<=(timedep+exclude_range) and y_E[i]==1:
                tmp_e.append(1)
                tmp_pred.append(pred_score[i])
                tmp_t.append(y_T[i])
            elif y_T[i]>timedep:
                tmp_e.append(0)
                tmp_pred.append(pred_score[i])
                tmp_t.append(y_T[i])

        tmp_pred = np.array(tmp_pred)
        tmp_e = np.array(tmp_e)
        tmp_t = np.array(tmp_t)

        if (~np.isfinite(tmp_pred)).any():
            tmp_pred = np.clip(tmp_pred,a_max=np.nanmax(tmp_pred))
        if np.count_nonzero(tmp_e)>0:
            try:
                auc_timesep = roc_auc_score(tmp_e,tmp_pred)
                fpr, tpr, thresholds = roc_curve(tmp_e, tmp_pred)
            except Exception as e:
                print('================',e,'==========')
                time_dep_auc_all[f'{timedep}_auc'] = np.nan
        else:
            auc_timesep = np.nan
            fpr, tpr, thresholds = None,None,None

        time_dep_auc_all[f'{timedep}_auc'] = auc_timesep
    return time_dep_auc_all

def make_riskset(time: np.ndarray) -> np.ndarray:
    """Compute mask that represents each sample's risk set.

    Parameters
    ----------
    time : np.ndarray, shape=(n_samples,)
        Observed event time sorted in descending order.

    Returns
    -------
    risk_set : np.ndarray, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set



class FeaDataset(Dataset):
    def __init__(
            self,
            data,
            T,
            E,
            ID=None,
            fea_params=None
        ) -> None:
       
        self.data = data
        self.T = T
        self.E = E
        self.ID = ID
        self.fea_params = ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        return self.data[indx], self.T[indx], self.E[indx], self.ID[indx]



def get_ont_hot(a,labels,is_onehot=False):
    if is_onehot:
        if len(labels) == 2:
            r = np.zeros((len(a),1))
            for i in range(len(a)):
                r[i,0] = labels.index(a[i])
        else:
            r = np.zeros((len(a),len(labels)))
            for i in range(len(a)):
                r[i,labels.index(a[i])] = 1
    else:
        r = np.zeros((len(a),1))
        for i in range(len(a)):
            r[i,0] = labels.index(a[i])        
    return r

def make_feature(df,use_fea=[]):
    feas = []
    for c,ll in {
                    'n_tumors':[1,2],
                    'pTNM':['0 / Tis','ⅠA','ⅡA','ⅡB','ⅢA','ⅢB','ⅢC'],
                    'HER2_status':[1,2],
                    'subtype':['Luminal B','Luminal A','TN','Her-2+'],
                    'T_stage':['T1','T2','T3','T4'],
                    'N_stage':['N0','N1','N2','N3'],
                    }.items():
        if c not in use_fea: continue
        fea_val = df[c].values
        try:
            ff = get_ont_hot(fea_val,ll)
        except Exception as e:
            print(c,ll)
            raise e
        feas.append(ff)
    for c in ['age',
                'ER_expression',
                'PR_expression',
                'Ki67_expression',
                'risk_score'
                ]:
        if c not in use_fea: continue
        ff = np.expand_dims(df[c].values,-1)*(-1)
        feas.append(ff)
    feas = np.concatenate(feas,1)

    return feas



def eval_ds(model,data_loader,cohort_name, device, tmp_result):
    model.eval()
    val_outputs_list, status_list, time_list, ID_list = [], [], [], []
    
    with torch.no_grad():
        for val_data in data_loader:
            inputs, dfs_time, status, ID = val_data       
            inputs = inputs.to(device).float()
            priors = torch.ones(inputs.shape[0], inputs.shape[1]).to(device)
            val_outputs,sparce_loss,mask = model(inputs,priors)
            
            val_outputs_list.append(val_outputs)
            status_list.append(status)
            time_list.append(dfs_time)
            ID_list.extend(ID)

        val_outputs = torch.cat(val_outputs_list,0)
        status = torch.cat(status_list,0)
        dfs_time = torch.cat(time_list,0)
        ID = ID_list

        val_c_index = c_index(val_outputs, dfs_time, status)

        aucs_val = get_timeDependent_auc(dfs_time.numpy(), 
                            status.numpy(), 
                            val_outputs.detach().cpu().numpy(), 
                            times=[12,24,36,48,60])

        tmp_result[f'{cohort_name}_c_index'] = val_c_index
        tmp_result[f'{cohort_name}_auc_1y'] = aucs_val['12_auc']
        tmp_result[f'{cohort_name}_auc_2y'] = aucs_val['24_auc']
        tmp_result[f'{cohort_name}_auc_3y'] = aucs_val['36_auc']
        tmp_result[f'{cohort_name}_auc_4y'] = aucs_val['48_auc']
        tmp_result[f'{cohort_name}_auc_5y'] = aucs_val['60_auc']

        return tmp_result,val_outputs.detach().cpu().numpy(),status.numpy(),dfs_time.numpy(),ID #,mask



train_fea = make_feature(train_df,all_feas)
train_T, train_E, train_ID  = train_df['time'].values, train_df['status'].values, train_df['ID'].values
test_fea = make_feature(test_df,all_feas)
test_T, test_E, test_ID = test_df['time'].values, test_df['status'].values, test_df['ID'].values





device = torch.device("cuda")
model = pytorch_tabnet.TabNet(inp_dim=train_fea.shape[1], out_dim=1, n_d=n_d, n_a=n_a, n_shared=2, n_ind=2, n_steps=n_steps, relax=relax, vbs=128).to(device)
model = nn.DataParallel(model)

batch_size = 64


train_ds = FeaDataset(train_fea,train_T,train_E,train_ID)
train_data_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False,num_workers=2)

test_ds = FeaDataset(test_fea,test_T,test_E,test_ID)
test_data_loader = DataLoader(test_ds, batch_size=len(test_fea), shuffle=False, pin_memory=False,num_workers=2)



lr = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr)
loss_function  = CoxPHLoss.coxPHLoss

best_ci = 0
model_best_epoch = 0
for epoch in range(1000):
    print('epoch:',epoch,'model_best_epoch:',model_best_epoch,best_ci)
    model.train()
    train_outputs_list, status_list, time_list = [], [], []
    for batch_id, batch_data in enumerate(train_data_loader):
        step_start = time.time()
        inputs, dfs_time, status, _ = batch_data    
        if inputs.shape[0]<2: continue       
        status_list.append(status)
        time_list.append(dfs_time)

        inputs = inputs.to(device).float()
        riskset = make_riskset(dfs_time.numpy())
        riskset = torch.tensor(riskset).to(device)
        optimizer.zero_grad()
        priors = torch.ones(inputs.shape[0], inputs.shape[1]).to(device)
        print('inputs',inputs.shape, priors.shape)
        outputs,sparse_loss,mask = model(inputs,priors)
        train_outputs_list.append(outputs)
        
        try:
            loss = loss_function(outputs, [status.to(device), riskset])
        except Exception as e:
            print(e)
            continue
        loss.backward()
        optimizer.step()
        print(f"train_loss: {loss.item():.4f}, step time: {(time.time() - step_start):.4f}")

    train_outputs_list = torch.cat(train_outputs_list,0)
    status = torch.cat(status_list,0)
    dfs_time = torch.cat(time_list,0)  
    train_c_index = c_index(train_outputs_list, dfs_time, status)


    if epoch>=4:
        tmp_result ={
            'n_d': n_d,
            'n_steps': n_steps,
            'relax':relax,
            'batch_size':batch_size,
            'lr':lr
        }

        tmp_result,train_score,train_e,train_t,train_id = eval_ds(model,train_data_loader,'train', device, tmp_result)
        tmp_result,test_score,test_e,test_t,test_id = eval_ds(model,test_data_loader,'test', device, tmp_result)

        ci_val = tmp_result['test_c_index']
           

        if best_ci < ci_val: 
            model_best_epoch = epoch
            best_ci = ci_val 
            model_ckpt_name = f'{epoch}_{ci_val:.5f}.pth'
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_root, model_ckpt_name),
            )   
            tmp_result['model_ckpt_name'] = model_ckpt_name
            tmp_result['ci_mean'] = ci_val
            tmp_result['epoch'] = epoch

            with open(os.path.join(ckpt_root, model_ckpt_name+'.pkl'),'wb') as f:
                pickle.dump({
                    'train':[train_score,train_t,train_e,train_id],
                    'test':[test_score,test_t,test_e,test_id],
                },f)


        print(tmp_result)
        if (epoch-model_best_epoch) > 30:
            break
    


            

'''
7343  74 76 68
'''