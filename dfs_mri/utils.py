import numpy as np
import sys
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score,roc_curve

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
    try:
        r = concordance_index(y, risk_pred, e)
    except Exception as e:
        print('No admissable pairs in the dataset.')
        r = 0.5
    return r


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
            # tmp_pred[~np.isfinite(tmp_pred)] = np.nan
            # tmp_pred[np.where(np.isnan(tmp_pred))] = np.nanmax(tmp_pred)*1.01
        if np.count_nonzero(tmp_e)>0:
            try:
                auc_timesep = roc_auc_score(tmp_e,tmp_pred)
                fpr, tpr, thresholds = roc_curve(tmp_e, tmp_pred)
            except Exception as e:
                # print(tmp_pred)
                print('================',e,'==========')
                time_dep_auc_all[f'{timedep}_auc'] = np.nan
            
        else:
            auc_timesep = np.nan
            fpr, tpr, thresholds = None,None,None

        time_dep_auc_all[f'{timedep}_auc'] = auc_timesep
    return time_dep_auc_all