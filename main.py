import os

import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold

from DataLoader_num import FeatureDictionary, DataParser

from matplotlib import pyplot as plt


import config

from metrics import gini_norm

from DNN_PNN import PNN


import random

def load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTrain.rename(columns={'Unnamed: 0':'id1'}, inplace = True)
    print(dfTrain.columns)
    # dfTrain = dfTrain[:200].copy()

    dfTest = pd.read_csv(config.TEST_FILE)
    dfTest.rename(columns={'Unnamed: 0':'id1'}, inplace = True)
    print(dfTest.columns)
    # dfTest = dfTest[:200].copy()


    def preprocess(df):
        #print(df)
        cols = [c for c in df.columns if c not in ['id','id1','target']]

        #df['missing_feat'] = np.sum(df[df[cols]==-1].values,axis=1)

        #df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)

#        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

        return df



    dfTrain = preprocess(dfTrain)

    dfTest = preprocess(dfTest)



    cols = [c for c in dfTrain.columns if c not in ['id','id1','target']]

    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    # #######自己加的随机选取10000进行测试###############
    # random.seed(2019)
    # cols = random.sample(cols,10000)
    # print(cols)



    X_train = dfTrain[cols].values

    y_train = dfTrain['target'].values

    print("****************y_train***************")

    print(y_train)



    X_test = dfTest[cols].values

    ids_test = dfTest['id'].values

    print("********************ids_test*******************")

    print(ids_test)



    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]



    return dfTrain,dfTest,X_train,y_train,X_test,ids_test,cat_features_indices



def run_base_model_dpnn(dfTrain,dfTest,folds,dpnn_params):
    #### 将输入文件转换为特征字典
    fd = FeatureDictionary(dfTrain,

                           dfTest,

                           numeric_cols=config.NUMERIC_COLS,

                           ignore_cols = config.IGNORE_COLS,

                           cate_cols = config.CATEGORICAL_COLS)

    print(fd.feat_dim)

    print(fd.feat_dict)


    #### 将特征字典生成xi,xv数据
    data_parser = DataParser(feat_dict= fd)

    # Xi_train ：列的序号

    # Xv_train ：列的对应的值

    cate_Xi_train, cate_Xv_train, numeric_Xv_train,y_train = data_parser.parse(df=dfTrain, has_label=True)

    # pd.DataFrame(cate_Xi_train).info()
    # pd.DataFrame(cate_Xv_train).info()
    # pd.DataFrame(numeric_Xv_train).info()
    # pd.DataFrame(y_train).info()

    cate_Xi_test, cate_Xv_test, numeric_Xv_test,ids_test = data_parser.parse(df=dfTest)


    # pd.DataFrame(cate_Xi_test).info()
    # pd.DataFrame(cate_Xv_test).info()
    # pd.DataFrame(numeric_Xv_test).info()
    # pd.DataFrame(y_test).info()
    # print(dfTrain.dtypes)
    print("***********************data_parser:**************************")
    print(pd.DataFrame(cate_Xi_train).head())
    print(pd.DataFrame(cate_Xv_train).head())
    print(pd.DataFrame(numeric_Xv_train).head())
    print("y_train:")
    print(y_train)
    print("ids_test:")
    print(ids_test)


    # print(fd.feat_dim)

    dpnn_params["cate_feature_size"] = fd.feat_dim

    dpnn_params["field_size"] = len(cate_Xi_train[0])

    dpnn_params['numeric_feature_size'] = len(config.NUMERIC_COLS)

    print(dpnn_params["cate_feature_size"])

    print(dpnn_params["field_size"])

    print(dpnn_params['numeric_feature_size'])

    ### 初始化y_train和y_test
    y_train_meta = np.zeros((dfTrain.shape[0],1),dtype=float)

    y_test_meta = np.zeros((dfTest.shape[0],1),dtype=float)

    print("y_meta:***************************************")

    print(y_train_meta)

    print(y_test_meta)



    _get = lambda x,l:[x[i] for i in l]


     ##初始化gini result
    gini_results_cv = np.zeros(len(folds),dtype=float)

    gini_results_epoch_train = np.zeros((len(folds),dpnn_params['epoch']),dtype=float)

    gini_results_epoch_valid = np.zeros((len(folds),dpnn_params['epoch']),dtype=float)



    for i, (train_idx, valid_idx) in enumerate(folds):




        ###训练集
        cate_Xi_train_, cate_Xv_train_, numeric_Xv_train_,y_train_ = _get(cate_Xi_train, train_idx), _get(cate_Xv_train, train_idx),_get(numeric_Xv_train, train_idx), _get(y_train, train_idx)

        cate_Xi_valid_, cate_Xv_valid_, numeric_Xv_valid_,y_valid_ = _get(cate_Xi_train, valid_idx), _get(cate_Xv_train, valid_idx),_get(numeric_Xv_train, valid_idx), _get(y_train, valid_idx)

        pd.DataFrame(cate_Xi_train_).info()

        pd.DataFrame(cate_Xi_valid_).info()

        print("y_train_:")

        print(y_train_)

        print("y_valid_:")

        print(y_valid_)



        # print(dpnn_params)

        dpnn = PNN(**dpnn_params)


        dpnn.fit(cate_Xi_train_, cate_Xv_train_, numeric_Xv_train_,y_train_, cate_Xi_valid_, cate_Xv_valid_, numeric_Xv_valid_,y_valid_)

        dpnn.save_result(i)



        y_train_meta[valid_idx,0] = dpnn.predict(cate_Xi_valid_, cate_Xv_valid_, numeric_Xv_valid_)

        y_test_meta[:,0] += dpnn.predict(cate_Xi_test, cate_Xv_test, numeric_Xv_test)


        ###测试集的gini系数
        #print("y_train_meta")
        #print(y_train_meta[valid_idx][:,0])
        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx][:,0])
        # fold_filename = str(i) + "gini_results_cv.csv"
        # pd.DataFrame(gini_results_cv[i]).to_csv(fold_filename)

        gini_results_epoch_train[i] = dpnn.train_result
        fold_filename1 = str(i) + "gini_results_epoch_train.csv"
        pd.DataFrame(gini_results_epoch_train[i]).to_csv(os.path.join(config.SUB_DIR, fold_filename1))

        gini_results_epoch_valid[i] = dpnn.valid_result
        fold_filename2 = str(i) + "gini_results_epoch_valid.csv"
        pd.DataFrame(gini_results_epoch_valid[i]).to_csv(os.path.join(config.SUB_DIR, fold_filename2))

    pd.DataFrame(gini_results_cv).to_csv(os.path.join(config.SUB_DIR, "gini_results_cv.csv"))


    y_test_meta /= float(len(folds))



    # save result

    if dpnn_params["use_pnn"] and dpnn_params["use_deep"]:

        clf_str = "DNN-PNN"

    elif dpnn_params["use_pnn"]:

        clf_str = "pnn"

    elif dpnn_params["use_deep"]:

        clf_str = "PNN"

    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))

    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())

    _make_submission(ids_test, y_test_meta, filename)

    print(gini_results_epoch_train)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)



    return y_train_meta, y_test_meta



def _make_submission(ids, y_pred, filename="submission.csv"):

    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(

        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")



def _plot_fig(train_results, valid_results, model_name):

    colors = ["red", "blue", "green", "yellow", "purple"]

    xs = np.arange(1, train_results.shape[1]+1)

    plt.figure()

    legends = []

    for i in range(train_results.shape[0]):
        print(train_results[i])


        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")

        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")

        legends.append("train-%d"%(i+1))

        legends.append("valid-%d"%(i+1))

    plt.xlabel("Epoch")

    plt.ylabel("Pearsonr")

    plt.title("%s"%model_name)

    plt.legend(legends)

    plt.savefig(os.path.join(config.SUB_DIR, "fig/%s.png"%model_name))

    plt.close()


###设置dnnpnn的参数
dnnpnn_params = {

    "use_pnn":True,##dnnpnn中的pnn部分

    "use_deep":True,
    "embedding_size":8,

    "dropout_pnn":[1.0,1.0],

    "deep_layers":[35,35],

    "dropout_deep":[0.9,0.9,0.9],

    "deep_layer_activation":tf.nn.relu,

    "epoch":25,

    "batch_size":200,

    "learning_rate":0.001,

    "optimizer":"adam",

    "batch_norm":1,

    "batch_norm_decay":0.995,

    "l2_reg":0.2,#0.2

    "verbose":True,

    "eval_metric":gini_norm,

    "random_seed":config.RANDOM_SEED

}



# load data

dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()
print("dfTrain")
print(dfTrain)
print("dfTest")
print(dfTest)

dfTrain.info()
dfTest.info()

# folds

#folds = list(StratifiedKFold(n_splits=config_zhuanli.NUM_SPLITS, shuffle=True,（用于分类）
#用于回归
folds = list(KFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))


print(folds)
#y_train_dpnn,y_test_dpnn = run_base_model_dpnn(dfTrain,dfTest,folds,dpnn_params)

y_train_dpnn, y_test_dpnn = run_base_model_dpnn(dfTrain, dfTest, folds, dnnpnn_params)



# # ------------------ PNN Model ------------------
#
# pnn_params = dpnn_params.copy()
#
# pnn_params["use_deep"] = False
#
# y_train_pnn, y_test_pnn = run_base_model_dpnn(dfTrain, dfTest, folds, pnn_params)



# # ------------------ DNN Model ------------------
#
# dnn_params = dpnn_params.copy()
#
# dnn_params["use_pnn"] = False
#
# y_train_dnn, y_test_dnn = run_base_model_dpnn(dfTrain, dfTest, folds, dnn_params)



############使用DeepPNN_BN_NEW##################
#pnn+DNN+product
#DNN+Product
