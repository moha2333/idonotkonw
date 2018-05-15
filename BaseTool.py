__author__ = 'jmh081701'
import random
import numpy as np
import tensorflow as tf 
from scipy import sparse
class Preprocess():
    def __init__(self,train_x,train_y,test_x,splitratio=0.1):
        samplenum=train_x.shape[0]
        end=int(samplenum*splitratio)
        _train_y=list()
        data=list()
        row=list()
        col=list()
        #for i in range(len(train_y)):
        #    if train_y[i]==1:
        #       data.append(1.0)
        #       row.append(i)
        #       col.append(1)
        #train_y=sparse.coo.coo_matrix((data,(row,col)),shape=(samplenum,2))
        for i in range(len(train_y)):
            if train_y[i]==1:
               _train_y.append([0.,1.0])
            else:
               _train_y.append([1.0,0.])
        train_y=_train_y
        print(end,train_x.shape)
        self.train_x=train_x.tocsr()[end:,]
        #self.train_y=train_y.tocsr()[end:]
        self.train_y=train_y[end:]
        

        self.valid_x=train_x.tocsr()[:end,]
        #self.valid_y=train_y.tocsr()[:end]
        self.valid_y=train_y[:end]
      
        self.test_x=test_x
        self.used_train=set()
        self.used_valid=set()
        self.veclen=self.train_x.shape[1]
        self.train_size=self.train_num()
        self.valid_size=self.valid_num()
        print("out")
        #del train_x
    def next_train_batch(self,batchSize=100000):
        batchsize=batchSize
        #print(len(self.train_y))
        index= random.randint(0,int(len(self.train_y)/batchsize)-1)
        #print(index)
        print(index*batchsize,(index+1)*batchsize)
        tmpMat=(self.train_x.tocsr()[index*batchsize:(index+1)*batchsize,]).tocoo()
        #tf_coo_matrix=tf.SparseTensorValue(indices=np.array([tmpMat.row,tmpMat.col]).T,values=tmpMat.data,dense_shape=tmpMat.shape)
        tf_coo_matrix=tmpMat.toarray()
        #tmpMaty=self.train_y.tocsr()[index*batchsize:(index+1)*batchsize,]
        #tf_coo_matrixy=tf.SparseTensorValue(indices=np.array([tmpMaty.row,tmpMaty.col]).T,values=tmpMaty.data,dense_shape=tmpMaty.shape)
        tf_coo_matrixy=self.train_y[index*batchsize:(index+1)*batchsize]
        return tf_coo_matrix,tf_coo_matrixy

    def next_valid_batch(self,batchSize=100000):
        batchsize=batchSize
        index= random.randint(0,int(len(self.valid_y)/batchsize)-1)
        tmpMat=(self.valid_x.tocsr()[index*batchsize:(index+1)*batchsize,]).tocoo()
        #tf_coo_matrix=tf.SparseTensorValue(indices=np.array([tmpMat.row,tmpMat.col]).T,values=tmpMat.data,dense_shape=tmpMat.shape)
        tf_coo_matrix=tmpMat.toarray()
        #tmpMaty=self.valid_y.tocsr()[index*batchsize:(index+1)*batchsize,]
        #tf_coo_matrixy=tf.SparseTensorValue(indices=np.array([tmpMaty.row,tmpMaty.col]).T,values=tmpMaty.data,dense_shape=tmpMaty.shape)
        tf_coo_matrixy=self.valid_y[index*batchsize:(index+1)*batchsize]
        return tf_coo_matrix,tf_coo_matrixy
    
    def test_vectors(self):
        tf_coo_matrix=tf.SparseTensorValue(indices=np.array([self.test_x.row,self.test_y.col]).T,values=self.test_x.data,dense_shape=self.test_x.shape)
        return tf_coo_matrix

    def test_labels(self):
        #tmpMat=sparse.coo.coo_matrix(([],([],[])),shape=(self.test_x.shape[0],2))
        #tf_coo_matrixy=tf.SparseTensorValue(indices=np.array([tmpMaty.row,tmpMaty.col]).T,values=tmpMaty.data,dense_shape=tmpMaty.shape)
        tf_coo_matrixy=np.zeros(shape=(self.test_x.shape[0],2))
        return tf_coo_matrixy
    def train_num(self):
        return self.train_x.shape[0]
    def valid_num(self):
        return self.valid_x.shape[0]

    def debug(self):
        pass
