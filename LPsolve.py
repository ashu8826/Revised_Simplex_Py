import numpy as np
import LPdata as data
import RSM

#first make b>= 0 then find bv then x 
bv = [-1]*len(data.A)
x = []
y=[]
E = []
c = []
non_arti_conts = []
arti_conts = []
noarti = 0
ret = None
for i in range(len(data.b)):
    if(data.b[i]<0):
        data.b[i] = -1*data.b[i]
        data.A[i] = [-1*x for x in data.A[i]]
m = len(data.A)
data.A = np.array(data.A)

for i in range(data.A.shape[1]):  #column
    c.append(0)
    cons = -1
    bvn = -1 
    for j in range(len(data.A)): #row
        if(cons == -1 and data.A[j][i]>0):
            cons = j
            bvn = i
        elif (data.A[j][i]<0) or (cons != -1 and data.A[j][i]!=0):
            bvn = -1
            break
    if bvn != -1:
        data.b[cons] = data.b[cons]/data.A[cons][bvn]
        for i in range(len(data.A[cons])):
            data.A[cons][i] = data.A[cons][i]/data.A[cons][bvn]
        bv[cons] = bvn
        x.append(data.b[cons])
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
    toadd = []
    I = np.diag([1]*noarti)
    k=0
    for i in range(len(data.A)):
        if i in non_arti_conts:
            toadd.append([0]*noarti)
        elif i in arti_conts:
            x.append(data.b[i])
            bv[i] = len(x)-1
            c.append(-1)
            toadd.append(I[k])
            k = k+1
    y = [c[i] for i in bv]
    data.A = np.append(data.A, np.array(toadd), axis=1)
    
    #print("A",data.A)
    #print("b",data.b)
    #print("c",c)
    #print("x",x)
    #print("y",y)
    #print("bv",bv)
    rsm1 = RSM.RSM(data.A,data.b,c,x,y,bv,phase1,noarti)
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
    
    #print("A",data.A)
    #print("b",data.b)
    #print("c",data.c)
    #print("x",x)
    #print("bv",bv)
    #print("y",y)
    #print("E",E)
    #print("obj",getattr(rsm1,"obj"))

if ret!="infeasible":
    rsm2 = RSM.RSM(data.A,data.b,data.c,x,y,bv,phase1,noarti,E)
    rsm2.solve()
    print("x:  ",getattr(rsm2,'x'))
    print("y:  ",[round(x,10) for x in getattr(rsm2,'y')])
    print("obj:",getattr(rsm2,'obj'),"dual objective: ",rsm2.cal_dual_obj())
    