
"""
 进一步提取了用户兴趣和主题的特征，其余无变化，主要提取了interest1, 2, 5, topic1，其余未提取因为大部分比值为零而且本地测试效果下降
"""

import numpy as np
import pandas as pd
import os
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import  metrics
# import xgboost as xgb
import lightgbm as lgb
import pickle as pk
import math
import  tensorflow as tf
from  BaseTool import  *
import time
workspace ="./data/"
start=time.clock()
if not os.path.exists(workspace+'model_fea_add3_train_y.pkl'):
    # 下面根据前面的特征分析，剔除掉缺失率过高的特征: interest3, interest4, kw3, appIdInstall, topic3
    ad_feature=pd.read_csv(workspace+'adFeature.csv')
    user_feature = pd.read_csv(workspace+'userFeature.csv')

    one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
           'adCategoryId', 'productId', 'productType']
    vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']

    train = pd.read_csv(workspace+'train.csv')
    predict = pd.read_csv(workspace+'test1.csv')

    print('all data had been loaded!')
    train.loc[train['label']==-1,'label']=0
    predict['label']=-1
    data = pd.concat([train,predict])
    data = pd.merge(data,ad_feature,on='aid',how='left')
    data = pd.merge(data,user_feature,on='uid',how='left')
    data = data.fillna('-1')

    print('start!')
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]
    data_clicked = train[train['label'] == 1]


    # 增加每个广告推送给不同的用户数

    print('开始加入广告推送给不同用户的数特征')

    num_advertise_touser = train.groupby('aid').uid.nunique()
    num_advertise_touser = pd.DataFrame({
        'aid': num_advertise_touser.index,
        'num_advertise_touser' : num_advertise_touser.values
    })
    data = pd.merge(data, num_advertise_touser, on=['aid'], how='left')

    # 增加各种兴趣ID，和主题ID的比例值


    def get_common_interest(type_name, ratio):
        num_adid = data_clicked['aid'].value_counts().sort_index().index
        num_aid_clicked = dict(data_clicked['aid'].value_counts().sort_index())
        num_user_clicksameAd_interest = data_clicked.groupby('aid')[type_name].value_counts()
        dict_interest = {}
        for adid in num_adid:
            dict_buf = {}
            for interest in num_user_clicksameAd_interest.items():
                index = interest[0]
                if index[0] == adid:
                    number = interest[1]
                    detail = index[1]
                    detail = detail.split(' ')
                    for det in detail:
                        if det not in dict_buf:
                            dict_buf[det] = number
                        else:
                            dict_buf[det] += number
            dict_interest[adid] = dict_buf
        dict_common_interest = []
        for adid, dict_inter in dict_interest.items():
            dict_common_buf = {}
            dict_common_buf['aid'] = adid
            common_inter = []
            ad_total = num_aid_clicked[adid] - dict_inter.get('-1', 0)
            if '-1' in dict_inter:
                dict_inter.pop('-1')
            for id_inter, num in dict_inter.items():
                if num >= ad_total*ratio:
                    common_inter.append(id_inter)
            str_name = 'common_'+type_name
            dict_common_buf[str_name] = common_inter
            dict_common_interest.append(dict_common_buf)
        return dict_common_interest



    # 获取相同的兴趣ID2
    print('开始加入兴趣ID2')
    dict_common_interest2 = get_common_interest('interest2', 0.25)
    df_common_interest2 = pd.DataFrame(dict_common_interest2)
    data = pd.merge(data, df_common_interest2, on=['aid'], how='left')


    # 获取相同的兴趣ID5
    print('开始加入兴趣ID5')
    dict_common_interest5 = get_common_interest('interest5', 0.25)
    df_common_interest5 = pd.DataFrame(dict_common_interest5)
    data = pd.merge(data, df_common_interest5, on=['aid'], how='left')


    #获取相同的兴趣ID1
    print('开始加入兴趣ID1')
    dict_common_interest1 = get_common_interest('interest1', 0.25)
    df_common_interest1 = pd.DataFrame(dict_common_interest1)
    data = pd.merge(data, df_common_interest1, on=['aid'], how='left')

    # 获取相同的主题1
    print('开始加入主题1')
    dict_common_topic1 = get_common_interest('topic1', 0.1)
    df_common_topic1 = pd.DataFrame(dict_common_topic1)
    data = pd.merge(data, df_common_topic1, on=['aid'], how='left')

    data['num_common_topic1'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['topic1', 'common_topic1']].values]
    data['num_common_interest1'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest1', 'common_interest1']].values]
    data['num_common_interest2'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest2', 'common_interest2']].values]
    data['num_common_interest5'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest5', 'common_interest5']].values]
    # 分离测试集
    data = data.fillna(0)
    # 将正样本重复采样
    tmp = data[data.label == 1]
    for i in range(10):
        data = pd.concat([data, tmp])
    # 将数据打乱
    data.sample(frac=1).reset_index(drop=True)

    train = data[data.label != -1]

    test = data[data.label == -1]
    res = test[['aid','uid']]
    test = test.drop('label', axis=1)
    train_y = train.pop('label')

    pk.dump(train_y, open(workspace+'model_fea_add3_train_y.pkl', 'wb'), protocol=4)
    pk.dump(res, open(workspace+'model_fea_add3_test.pkl', 'wb'), protocol=4)
    # 处理联网类型特征
    ct_train = train['ct'].values
    ct_train = [m.split(' ') for m in ct_train]
    ct_trains = []
    for i in ct_train:
        index = [0, 0, 0, 0, 0]
        for j in i:
            index[int(j)] = 1
        ct_trains.append(index)

    ct_test = test['ct'].values
    ct_test = [m.split(' ') for m in ct_test]
    ct_tests = []
    for i in ct_test:
        index = [0, 0, 0, 0, 0]
        for j in i:
            index[int(j)] = 1
        ct_tests.append(index)


    # 将上面新加入的特征进行归一化
    print('归一化...')
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(data[['num_advertise_touser', 'num_common_interest1', 'num_common_interest2',
                     'num_common_interest5', 'num_common_topic1']].values)
    train_x = scaler.transform(train[['num_advertise_touser', 'num_common_interest1', 'num_common_interest2',
                                      'num_common_interest5', 'num_common_topic1']].values)

    test_x = scaler.transform(test[['num_advertise_touser', 'num_common_interest1', 'num_common_interest2',
                                    'num_common_interest5', 'num_common_topic1']].values)
    train_x = np.hstack((train_x, ct_trains))
    test_x = np.hstack((test_x, ct_tests))


    # 特征进行onehot处理
    enc = OneHotEncoder()

    oc_encoder = OneHotEncoder()
    for feature in one_hot_feature:
        oc_encoder.fit(data[feature].values.reshape(-1, 1))
        train_a=oc_encoder.transform(train[feature].values.reshape(-1, 1))
        test_a = oc_encoder.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('one-hot prepared !')

    # 处理count特征向量

    ct_encoder = CountVectorizer(min_df=0.0009)
    for feature in vector_feature:
        ct_encoder.fit(data[feature])
        train_a = ct_encoder.transform(train[feature])
        test_a = ct_encoder.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('cv prepared !')
    # print('ths shape of train data:', test_x.shape)
    sparse.save_npz(workspace+'model_fea_add3_train.npz', train_x)
    sparse.save_npz(workspace+'model_fea_add3_test.npz', test_x)
else:
    print('load data from dump file.....')
    train_x=sparse.load_npz(workspace+'model_fea_add3_train.npz')
    test_x=sparse.load_npz(workspace+'model_fea_add3_test.npz')
    train_y=pk.load(open(workspace+'model_fea_add3_train_y.pkl',"rb"))
    res=pk.load(open(workspace+'model_fea_add3_test.pk',"rb"))
print("data prepare well")
end=time.clock()
print("using  %s s"%(end-start))
print("train_x,train_y,test_x,res",type(train_x),type(train_y),type(test_x),type(res))
print(train_x.shape,train_y.shape)
print(test_x.shape,res.shape)

_train_x=train_x
_train_y=list(train_y)
_test_x=test_x
data=Preprocess(_train_x,_train_y,_test_x)
InputColumn=7
InputRow=int(5299/InputColumn)
#InputX=tf.sparse_placeholder(dtype=tf.float32,shape=[None,None],name="InputX")
InputX=tf.placeholder(dtype=tf.float32,shape=[None,None],name="InputX")
InputY=tf.placeholder(dtype=tf.float32,shape=[None,2],name="InputY")

'''
    输入InputX是原始输入,其中InputX[i]表示第i个样本的特征向量。
    输入InputY是原始的label. InputY[i]表示第i个训练样本的期望label.
'''

dir="./"

writer=tf.summary.FileWriter(dir+".//cnngrahph",tf.get_default_graph())
merged=tf.summary.merge_all()

'''
    Saver:模型保存器
'''

with tf.name_scope('C1'):
    W_C1=tf.Variable(tf.truncated_normal([3,3,1,32],stddev=0.01),dtype=tf.float32)
    b_C1=tf.Variable(tf.constant(0.1,tf.float32,shape=[32]))
    #W_C1是C1层的权值矩阵,它也是卷积核，共有32个卷积核。
    # b_C1则是偏置
    print("reshape.....")
    #InputX=tf.sparse_tensor_to_dense(InputX)
    X=tf.reshape(InputX,[-1,InputRow,InputColumn,1])
    print(X.shape)
    #需要对输入转化为conv2d想要的格式
    featureMap_C1=tf.nn.conv2d(X,W_C1,[1,1,1,1],padding='SAME')+b_C1
    #conv2d的参数：
    #input:[图片个数,图片长，图片宽，图片的通道数]
    #filter:[滤波器长，滤波器宽，输入通道数，输出通道数]
    #stride:[1,1,1,1] 在四个轴上跳跃的大小
    #OK,C1卷积完成

with tf.name_scope('f'):
    relu_C1=tf.nn.relu(featureMap_C1)  #激活层
with tf.name_scope('S2'):
    featureMap_S2=tf.nn.max_pool(relu_C1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #S2的池化。
with tf.name_scope('C3'):
    W_C3=tf.Variable(tf.truncated_normal([3,3,32,64],stddev=0.01))
    b_C3=tf.Variable(tf.constant(0.1,tf.float32,shape=[64]))
    featureMap_C3=tf.nn.conv2d(featureMap_S2,W_C3,[1,1,1,1],padding='SAME')+b_C3

with tf.name_scope('f'):
    relu_C3=tf.nn.relu(featureMap_C3)
with tf.name_scope('S4'):
    featureMap_S4=tf.nn.max_pool(relu_C3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#C3以及S4的过程
with tf.name_scope('flat'):
    per=int(math.ceil(InputColumn/4.0)*math.ceil(InputRow/4.0) *64)
    fetureMap_flatter=tf.reshape(featureMap_S4,[-1,per])
#栅格化
with tf.name_scope('fullcont'):
    W_F5=tf.Variable(tf.truncated_normal([int(fetureMap_flatter.shape[1]),512],stddev=0.1))
    b_F5=tf.Variable(tf.constant(0.1,tf.float32,shape=[512]))
    out_F5=tf.nn.relu(tf.matmul(fetureMap_flatter,W_F5)+b_F5)
    #out_F5_drop=tf.nn.dropout(out_F5,keep_prob)
#全连接层完成
with tf.name_scope('output'):
    W_OUTPUT=tf.Variable(tf.truncated_normal([512,2],stddev=0.01))
    b_OUTPUT=tf.Variable(tf.constant(0.1,tf.float32,shape=[2]))
    predictY=tf.nn.softmax(tf.matmul(out_F5,W_OUTPUT)+b_OUTPUT,name="predictY")
outputY=tf.add(predictY,0,name="outputY")
#输出层,使用softmax函数

#loss=tf.reduce_mean(-tf.reduce_sum(InputY*tf.log(predictY)))
loss=tf.nn.softmax_cross_entropy_with_logits(labels=InputY,logits=predictY)
#tf.summary.histogram('loss',loss)
#tf.summary.scalar('loss',loss)
#残差函数loss设置为交叉熵
learning_rate=1e-4
#train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)
train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)

#y_pred=tf.arg_max(predictY,1)
#bool_pred=tf.equal(tf.arg_max(InputY,1),y_pred)
#right_rate=tf.reduce_mean(tf.to_float(bool_pred))
#tf.summary.scalar("right rate",right_rate)
Saver=tf.train.Saver()


def load_model(sess,modelname="cnnmodel"):
    ckpt=tf.train.get_checkpoint_state(dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("*"*30)
        print("load lastest model......")
        Saver.restore(sess,dir+modelname)
        print("*"*30)

def save_model(sess,modelname="cnnmodel"):
    print("*"*30)
    Saver.save(sess,dir+modelname)
    print("saving model well.")
    print("*"*30)
def to_csv(output,res,filename=workspace+"submission.csv"):
        res['score']=np.array(output)[:,1]
        res['score']=res['score'].apply(lambda x:float('%0.6'%x))
        res.to_csv(filename,index=False)
def cal_auc(real_y,predict_y):
    return metrics.roc_auc_score(real_y,predict_y)
#merge_op=None
#merge_op2=None
print("start....")
with tf.Session() as sess:
    print("begin....training......")
    init =tf.global_variables_initializer()
    sess.run(init)

    step=1
    sameMAX=40
    sameStep=0
    accSum=0
    batchsize=300
    batch_epoch=data.train_num()/batchsize
    load_model(sess)
    while True:
        #print(step)
        if(step%200==0):
            #测试一下
            test_vec,test_lab=data.next_valid_batch(batchSize=500)
            #tf.summary.scalar('valid:rate',right_rate)

            #if merge_op2==None:
            #    merge_op2=tf.summary.merge_all()
            out=sess.run([outputY],{InputX:test_vec,InputY:test_lab})[0]
            #writer.add_summary(summary_,step)
            #print(out,np.array(out)[:,1])
            print("#"*30)
            #print(np.array(test_lab)[:,1])
            auc=cal_auc(np.array(test_lab)[:,1],np.array(out)[:,1])
            print({"!!!!!!!!!!!!!!testing:"+str(step)+" auc":auc})
            accSum=accSum+auc
            sameStep+=1
            if(sameStep%sameMAX==0):
                if(auc==accSum/sameMAX):
                    print({step:auc})
                    break
                else:
                    accSum=0
                    sameStep=0
                save_model(sess)
            step=step+1
            continue
        train_vec,train_lab=data.next_train_batch(batchSize=batchsize)
        #print(type(train_vec),type(train_lab))
        #if merge_op==None:
        #    merge_op=tf.summary.merge_all()
        #np.array(train_vec)
        #print("lallalal",train_vec.shape)
        #np.array(train_lab)
        #print(train_lab)
        #np.array(train_lab)
        #print("2233")
        l,op=sess.run([loss,train_op],feed_dict={InputX:train_vec,InputY:train_lab})
        #py,l,op=sess.run([predictY,loss,train_op])
        print(step)
        if(step%20==0):
            #每隔20批,跟踪一次
            #writer.add_summary(summary,step)
            pass
        step=step+1
    save_model(sess)
    #print(data.test_vectors())
    output=sess.run([outputY],feed_dict={InputX:data.test_vectors(),InputY:data.test_labels()})[0]
    to_csv(output,res)
