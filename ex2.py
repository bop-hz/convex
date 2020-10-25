import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt 
my_data = np.genfromtxt('xy_train.csv',delimiter=',')
y=my_data[:,2]
x1=my_data[:,0]
x2=my_data[:,1]
my_data = np.delete(my_data,2,1)
C = 1
nn=200
p = 2
my_data[:,0]=np.multiply(my_data[:,0],y)
my_data[:,1] = np.multiply(my_data[:,1],y)
coremat = my_data @ (my_data.T)
w = cp.Variable(nn)
unit_vec = np.ones(nn)
cost  =-0.5*cp.quad_form(w,coremat)+unit_vec.T @ w
prob = cp.Problem(cp.Maximize(cost),[w>=0,w<=C*unit_vec,y.T @ w ==0])
prob.solve(solver=cp.SCS)
beta_vec = my_data.T @ (w.value)
print(beta_vec)