# In[23]:
#LP Maximization Instances
# unbounded example
A = [[2,-2,1,0],
     [4,0,0,1]]
c = [4,6,0,0]
b = [6,16]  
# In[23]:
# general example   
A = [[1,1,1,0,0],
     [2,3,0,1,0],
     [1,5,0,0,1]]
c = [6,8,0,0,0]
b = [10,25,35]

# In[23]:
#needs artifical variable
A = [[1,2,3,0],
     [-1,2,6,0],
     [0,4,9,0],
     [0,0,3,1]]
b = [3,2,5,1]
c = [1,1,1,0]
# In[23]:
import numpy as np

train_data = np.load("dataset\DB_Vecs.npy")
train_label = np.load("dataset\DB_Labels.npy")
test_data = np.load("dataset\Q_vecs.npy")
train_label = train_label.reshape((train_label.shape[0],1))

data_label = np.concatenate((train_data,train_label),axis = 1)
uni,unique_indices,unique_inverse,unique_counts = np.unique(data_label,return_index=True, return_inverse=True, return_counts=True, axis=0)
uni = uni[uni[:,-1].argsort()]

train_data = uni
train_data = np.delete(train_data, -1, 1)
train_label = np.array(uni[:,-1]).reshape((train_data.shape[0],1))

train_label[train_label==0] = -1
c = [0]*31 + [-1]*623 + [0]*623
b = [-1]*623

U = np.diag([-1]*623)
S = np.diag([1]*623)

one = np.array([[1]*623]).reshape((623,1))
A = train_label*np.concatenate((train_data,one),axis = 1)
A = np.concatenate((A,U),axis = 1)
A = np.concatenate((A,S),axis = 1)

