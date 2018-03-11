# ### Revised Simplex Algorithm for Maximization Problem.
import numpy as np
import sys

class RSM:
    'A = constraint matrix of size mxn\
     b = RHS of constraint b should be >= 0\
     c = cost coefficient of x\
     x = initial basic feasible solution\
     y = initial dual solution\
     bv= index of x in basis matrix\
     phase1 = true/false Phase 1 Lp or not \
     noarti = number of artifical variable added.\
     nbv = index of non basic variable\
     exit_index = index of exit variable\
     entry_index = index of entry variable\
     E = Eta matricses in array form\
     e_index = position wrt eta matrix which is exited\
     obj = objective function value\
     pbar = B^{-1}A_j of the entry variable required to find the ext variable'    
    def __init__(self,A,b,c,x,y,bv,phase1,noarti,E=[]):
        self.A = np.array(A,dtype='float').T  # A in transform form, A_1=A[1] A_2 A_3, quickly access for coefficient of a variable 
        self.b = np.array(b,dtype='float')
        self.c = np.array(c,dtype='float')
        self.x = np.array(x,dtype='float')
        self.y = np.array(y,dtype='float')
        self.bv = np.array(bv)
        self.nbv = np.array([z for z in range(len(self.x)) if z not in self.bv] )
        self.m = self.A.shape[1]
        self.obj = 0
        self.exit_index = 0
        self.entry_index = 0
        if E ==[]:  #initial E is first column of identity matrix
            self.E = []
            self.E.append([0]*(self.m+1))
            self.E[0][0] = 1
        else: # it will come here in phase 2. with E coming from phase 1
            self.E = E
        self.e_index = 0
        self.pbar = np.array([0]*self.m)
        self.phase1 = phase1
        self.noarti = noarti
        
    def cal_coeff(self,index): #c_j - z_j, if -ve for all nbv then optimal reached ,index of non-basic variable
        return self.c[index] - np.dot(self.y,np.transpose(self.A[index]))
    
    # reverse subsitution (backward direction)
    def p_solve(self,e,p):  #solve for pt in e.pt = p
        pt = [0]*len(p)
        pt[e[-1]] = p[e[-1]]/e[e[-1]]
        for i in [x for x in range(len(p)) if x != e[-1]]:
            pt[i] = p[i] - e[i]*pt[e[-1]]
        return pt
    
    def cal_pbar(self,p):  #solve for Pbar in E_1.E_2.Pbar_j = P_j by using E_2.Pbar_j = pt 
        pt = p
        for ei in self.E:
            pt = self.p_solve(ei,pt)
        return pt
    
    def y_solve(self,e,cb): #solve for yt in yt.e = cb 
        yt = cb
        s = 0
        for i in [x for x in range(len(cb)) if x != e[-1]]:
            s = s + yt[i]*e[i]
        yt[e[-1]] = (cb[e[-1]] - s)/e[e[-1]]
        return yt
    
    def cal_y(self,cb):  #solve for y in y.E_1.E_2 = cb by using y.E_1 = yt (y = B_{-1}c, B = E_1.E_2)
        yt = cb
        for e in reversed(self.E):
            yt = self.y_solve(e,yt)
        return yt
    
    def cal_obj(self):   #calculate objective value c^T.x
        return np.dot(self.x,self.c)
    
    def cal_dual_obj(self):   #calculate dual value b^T.y
        return np.dot(self.y,self.b)
    
    def changeEbvnbv(self):  #add new ETA matrix, change bv and nbv  
        e = [0]*(len(self.pbar)+1)
        e[-1] = self.e_index
        for i in range(len(self.pbar)):
            e[i] = self.pbar[i]
        self.E.append(e)
        self.bv[self.bv == self.exit_index] = self.entry_index
        self.nbv[self.nbv == self.entry_index] = self.exit_index
    
    def solve(self):   #main solver code
        while(True):
            
            cb = [self.c[i] for i in self.bv]   #calculate cb from initail bv
            self.y = self.cal_y(cb)           #calculate y 
            self.obj = self.cal_obj()           
            maximum = -1 #anti cycling using maximum coefficient rule and Bland Rule(smallest subscript rule)
            for i in self.nbv:    #calculating c_j-z_j for all nbv to decide the entry variable and optimality check
                coeff = self.cal_coeff(i) 
                if((coeff>maximum or (coeff == maximum and i<self.entry_index))and coeff>0):  #enforce Bland's rule for anticycling
                    maximum = coeff
                    self.entry_index = i
                    
            if(maximum == -1) and self.phase1:   #optimal reached and phase 1 is on
                if(self.obj!=0):        #if objective is not zero that means that original LP was infeasible
                    return "infeasible"
                
                index_arti = list(range(len(self.x)-self.noarti,len(self.x)))  #getting index of artifical variable in x, artifical variable are always in the end 
                artibv = np.intersect1d(self.bv, index_arti)  #getting arti vari which are still in the basis
                if(len(artibv)==0):
                    # remove artifical variable from the LP and start the phase 2 of RSM
                    self.A = np.delete(self.A,index_arti,0)
                    self.x = np.delete(self.x,index_arti)
                    print("Phase 1 complete")
                    return "feasible"
                else:
                    # kicking one AV out of basis. system is degenerate
                    self.exit_index = artibv[0]   #choosing first arti variable to be kicked out of the basis
                    self.e_index = int(np.where(self.bv == self.exit_index)[0]) #index of exit variable in self.bv
                    dependent = True
                    for i in self.nbv:
                        if i not in index_arti:
                            pnbv = self.cal_pbar(self.A[i]) #calculate pbar for all nbv which are not artifical
                            if pnbv[self.e_index]!= 0:      #anyone nbv have non zero value at e_index in the pbar 
                                self.pbar = pnbv            #than that is the entry variable
                                self.entry_index = i        #and system is degenerate.
                                dependent = False
                                break
                    
                    if dependent:   #system of linear equation(constraints) are not independent
                        print(self.e_index,"th constraint is dependent and is being removed")     #e_index th element in bv is AV and e_index th constraint is dependent (it is like e_index row in simplex table being delelted)
                        self.A = np.delete(self.A,self.e_index,1) 
                        for i in range(len(self.A)):
                            if np.count_nonzero(self.A[i])==0:
                                self.A = np.delete(self.A,i,0) 
                                break
                                
                        self.b = np.delete(self.b,self.e_index)
                        self.c = np.delete(self.c,self.exit_index)
                        self.x = [0]*len(self.c)
                        self.bv = [-1]*len(self.A.T)
                        self.y = []

                        #constructing in initial BFS for new (m-1) constraint by first choosing bv. Actually bv is not changed but because of deleting one variable and equation, indexes of variable are changed
                        for i in range(len(self.A)):
                            bvn  = -1
                            cons = -1
                            for j in range(len(self.A[i])):
                                if(cons == -1 and self.A[i][j]>0):
                                    cons = j
                                    bvn = i
                                elif (self.A[i][j]<0) or (cons != -1 and self.A[i][j]!=0):
                                    bvn = -1
                                    break
                        
                            if bvn != -1:
                                self.b[cons] = self.b[cons]/self.A[bvn][cons]
                                for i in range(len(self.A.T[cons])):
                                    self.A[i][cons] = self.A[i][cons]/self.A[bvn][cons]
                                self.bv[cons] = bvn

                        for i in range(len(self.bv)):
                            self.x[self.bv[i]] = self.b[i]
                        self.x = np.array(self.x)
                        self.bv = np.array(self.bv)
                        self.y = np.array([self.c[i] for i in self.bv])
                        self.nbv = np.array([z for z in range(len(self.x)) if z not in self.bv] )
                        self.m = self.A.shape[1]
                        self.obj = 0
                        self.exit_index = 0
                        self.entry_index = 0
                        self.E = []
                        self.E.append([0]*(self.m+1))
                        self.E[0][0] = 1
                        self.e_index = 0
                        self.pbar = np.array([0]*self.m)
                        self.phase1 = True
                        self.noarti = self.noarti - 1
                        continue
                        
                    else:  #system of equation are degenerate 
                        #print("not dependent")
                        theta = self.x[self.exit_index]/self.pbar[self.e_index]
                        self.x[self.entry_index] = theta
                        for i in range(self.m):
                            self.x[self.bv[i]] = self.x[self.bv[i]] - self.pbar[i]*theta
                        self.x[self.exit_index]  = 0.0
                        self.changeEbvnbv()
                        self.obj = self.cal_obj()
                        cb = [self.c[i] for i in self.bv]
                        self.y = self.cal_y(cb)
                    continue
                
            elif (maximum == -1) and not self.phase1: #optimal reached of phase 2
                return "feasible"
        
            self.pbar = self.cal_pbar(self.A[self.entry_index])
            #if all p are -ve then stop. problem is unbounded.
            if all(i <= 0 for i in self.pbar):
                print("problem is unbounded in direction of :",self.entry_index+1,"p:",self.pbar)
                return "unbounded"
        
            theta = sys.maxsize #have to be positive g
            for i in range(self.m):
                th = self.x[self.bv[i]]/self.pbar[i]
                if(((th < theta) or (th==theta and i<self.exit_index)) and th >0):  #blands's rule
                    theta = th
                    self.exit_index = self.bv[i]
                    self.e_index = i
            
            self.x[self.entry_index] = theta
            for i in range(self.m):
                self.x[self.bv[i]] = self.x[self.bv[i]] - self.pbar[i]*theta
            self.x[self.exit_index]  = 0.0

            self.changeEbvnbv()

            self.obj = self.cal_obj()
            cb = [self.c[i] for i in self.bv]
            self.y = self.cal_y(cb)        
            #checking for duality gap 
            #print("primal obj value: ",self.obj," dual obj value", self.cal_dual_obj())
    def Print(self):
        print("A",self.A)
        print("b",self.b)
        print("c",self.c)
        print("x",self.x)
        print("y",self.y)
        print("bv",self.bv)
        print("nbv",self.nbv)
        print("m",self.m)
        print("obj",self.obj)
        print("exit_index",self.exit_index)
        print("entry_index",self.entry_index)
        print("E",self.E)
        print("pbar",self.pbar)
        print("e_index",self.e_index)
        print("noarti",self.noarti)
        print()