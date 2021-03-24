'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-03-01 11:33:19
LastEditors: ZhangHongYu
LastEditTime: 2021-03-24 08:50:19
'''
import pandas as pd
from sklearn.model_selection import train_test_split

class FeatureDictionary(object):
    def __init__(self,trainfile=None,testfile=None,
                 dfTrain=None,dfTest=None,numeric_cols=[],
                 ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"

        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)

        else:
            dfTrain = self.dfTrain

        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)

        else:
            dfTest = self.dfTest

        df = pd.concat([dfTrain,dfTest])

        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1

            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us,range(tc,len(us)+tc)))
                # 后面是one-hot对应维度的列索引
                # [10 11  7  6  9  5  4  8  3  0  2  1 -1]
                # {10: 79, 11: 80, 7: 81, 6: 82, 9: 83, 5: 84, 4: 85, 8: 86, 3: 87, 0: 88, 2: 89, 1: 90, -1: 91}
                tc += len(us)

        self.feat_dim = tc


class DataParser(object):
    def __init__(self,feat_dict):
        self.feat_dict = feat_dict

    def parse(self,infile=None,df=None,has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"


        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            ye = dfi['emd_lable2'].values.tolist()
            dfi.drop(['seg_flight','emd_lable2'],axis=1,inplace=True)
        else:
            ids = dfi['seg_flight'].values.tolist()
            dfi.drop(['seg_flight'],axis=1,inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col,axis=1,inplace=True)
                dfv.drop(col,axis=1,inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()

        if has_label:
            return xi,xv,ye
        else:
            return xi,xv,ids


def data_preprocess(data):

    #  获得每个特征的缺失信息
    null_info = data.isnull().sum(axis=0)

    #  丢弃缺失值多于30%的特征
    features = [k for k, v in dict(null_info).items() if v < data.shape[0]* 0.3]
    data = data[features]

    null_info = data.isnull().sum(axis=0)

    # 选去出需要填补缺失值的特征
    features_fillna = [k for k, v in dict(null_info).items() if v > 0]

    # 对缺失值进行填补
    for feature in features_fillna:
        # 如果是非数值型特征或者是整型离散数值，用众数填补
        #将列按出现频率由高到低排序，众数即第一行，inplace表示原地修改
        if str(data[feature].dtype) == 'object' or str(data[feature].dtype) =='int64':
            data.loc[:,  feature] = data[feature].fillna(
                data[feature].mode().iloc[0]
            )
        #浮点连续数值型特征插值填补+平均数处理边缘
        else:
            #先将中间的数据插值处理
            data.loc[:,  feature] = data[feature].interpolate( method="zero", axis=0, limit_direction='both')
            #边缘直接填充平均数
            data.loc[:,  feature] = data[feature].fillna(
                data[feature].mean()
            )
    return data

def load_data():
    dfTrain = pd.read_csv('data/train.csv')
    X_submission = pd.read_csv('data/test.csv').drop('emd_lable2', axis=1)
    dfTrain = data_preprocess(dfTrain)
    X_submission = data_preprocess(X_submission)
    y_train = dfTrain['emd_lable2']
    X_train = dfTrain.drop('emd_lable2', axis=1)
    return dfTrain,X_train,y_train,X_submission