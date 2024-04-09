
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

def gini(actual, pred):
    #print("actual:",actual)
    #
    # assert (len(actual) == len(pred))
    #
    # all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    #
    # all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    #
    # totalLosses = all[:, 0].sum()
    #
    # giniSum = all[:, 0].cumsum().sum() / totalLosses
    #
    #
    #
    # giniSum -= (len(actual) + 1) / 2.
    #
    # return giniSum / len(actual)
    return pearsonr(actual, pred)[0]
    #return r2_score(actual, pred)

def gini_norm(actual, pred):

    # return gini(actual, pred) / gini(actual, actual)
    # print("actual_norm:",actual)
    # print("pred_norm:",pred)
     return pearsonr(actual, pred)[0]
    #return r2_score(actual, pred)
