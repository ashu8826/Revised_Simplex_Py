import numpy as np
import LPdata as data
import RSM

#first make b>= 0 then find bv then x 
bv = [-1]*len(data.A)
x = []
y=[]
E = []
c = []
non_arti_conts = []  #constraint number which dont require artifical variable
arti_conts = []     #constraint number which require artifical variable
noarti = 0           #no of artifical variable required
ret = None
for i in range(len(data.b)):   #making b>=0
    if(data.b[i]<0):
        data.b[i] = -1*data.b[i]
        data.A[i] = [-1*x for x in data.A[i]]
m = len(data.A)
data.A = np.array(data.A)

for i in range(data.A.shape[1]):  #column finding which constraint require artifical variable
    c.append(0)
    cons = -1   #constraint number
    bvn = -1    #index of basic variable number in x
    for j in range(len(data.A)): #row
        if(cons == -1 and data.A[j][i]>0):
            cons = j
            bvn = i
        elif (data.A[j][i]<0) or (cons != -1 and data.A[j][i]!=0):
            bvn = -1
            break
    if bvn != -1:
        data.b[cons] = data.b[cons]/data.A[cons][bvn]  #making coefficient in constrain of basic varibale to 1, so that we can easily find the intial basic feasible solution 
        for i in range(len(data.A[cons])):
            data.A[cons][i] = data.A[cons][i]/data.A[cons][bvn]
        bv[cons] = bvn         #basic variable while keeping their position correct accordingly 
        x.append(data.b[cons])   #initial basic feasible solution
        non_arti_conts.append(cons)
    else:
        x.append(0)

arti_conts = list(set(range(len(data.A)))-set(non_arti_conts))
        
if bv.count(-1)!=0:
    #print("artifical variable needed")
    noarti =bv.count(-1)
    phase1 = True
elif bv.count(-1)==0:
    #print("artifical variable not needed")
    y = [data.c[i] for i in bv]
    phase1 = False
        
if phase1:
    #add artifical variable noarti, chose bv, construct x,y and make auxillary LP and solve it
    #
    toadd = []   #coefficient of artifical variable in each constraint
    I = np.diag([1]*noarti)   #identity matrix
    k=0
    for i in range(len(data.A)):
        if i in non_arti_conts:
            toadd.append([0]*noarti)
        elif i in arti_conts:     #if ith constraint require artifical variable 
            x.append(data.b[i]) #then that arti is also in bv and BFS will be b of that constraint
            bv[i] = len(x)-1
            c.append(-1)    #cost function for auxillary LP
            toadd.append(I[k])
            k = k+1
    y = [c[i] for i in bv]   #dual variable for initial basis matrix (B = I)
    data.A = np.append(data.A, np.array(toadd), axis=1)

    rsm1 = RSM.RSM(data.A,data.b,c,x,y,bv,phase1,noarti)  #solving aux LP
    ret = rsm1.solve()
    data.A = np.array(getattr(rsm1,'A')).T
    data.b = getattr(rsm1,'b')
    x = getattr(rsm1,'x')
    E = getattr(rsm1,'E')
    y = getattr(rsm1,'y')
    bv = getattr(rsm1,'bv')
    phase1 = False
    noarti = 0
    print("Given problem is :",ret)
    
if ret!="infeasible":
    rsm2 = RSM.RSM(data.A,data.b,data.c,x,y,bv,phase1,noarti,E)  #solving original problem which Initial BFS as the BFS of aux LP with no artifical variable in it
    rsm2.solve()
    print("x:  ",getattr(rsm2,'x'))
    np.save("result/opt_x",x)
    print("y:  ",[round(x,10) for x in getattr(rsm2,'y')])
    print("obj:",getattr(rsm2,'obj'),"dual objective: ",rsm2.cal_dual_obj())


    
# In[23]:
#------Accurancy code---------------#
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
import random

test_data = np.load("dataset\Q_vecs.npy")
no_train_sample = test_data.shape[0]
no_variable = test_data.shape[1] + 1

one = np.array([random.gauss(1, 0.1) for i in range(no_train_sample)]).reshape((no_train_sample,1))
test_data = np.concatenate((test_data,one),axis = 1)

truth = np.load("result/labelsfortestdata.npy")
truth[truth==0] = -1

hyper = []
x = getattr(rsm2,'x')
for i in range(0,len(x[:(no_variable*2)]),2):
    hyper.append(x[i]-x[i+1])

np.save("result/opt_x_100",getattr(rsm2,'x'))   

pred = np.dot(test_data,hyper)
pred[pred>0] = 1
pred[pred<0] = -1
pred = -1 *pred
count = 0
for i in range(len(truth)):
    if truth[i]==pred[i]:
        count+=1
print(float(count/no_train_sample))

precision, recall, fscore, support = score(truth, pred)
print("Accuracy: ",count/no_train_sample)
print("f1 score: ",fscore)
print("precision score: ",precision)
print("recall score: ",recall)
print("support:  ",support)