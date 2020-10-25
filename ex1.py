import cvxpy as cp
import numpy as np
import matplotlib.pyplot as  plt 
my_data = np.genfromtxt('xy_train.csv',delimiter=',')
y=my_data[:,2]
x1=my_data[:,0]
x2=my_data[:,1]
my_data = np.delete(my_data,2,1)
C = 1
nn=200
p = 2
epsilion_vec = cp.Variable(nn)
beta0 = cp.Variable()
beta_vec = cp.Variable(p)
cost = 0.5*cp.sum_squares(beta_vec)+C*cp.sum(epsilion_vec)
prob = cp.Problem(cp.Minimize(cost),[epsilion_vec>=0,cp.multiply(y,my_data @ beta_vec+beta0)+epsilion_vec>=1])
prob.solve()

beta0 = beta0.value
beta_vec = beta_vec.value
print(beta_vec)
x= np.linspace(-3,4,num=50)
yy= -(beta_vec[0]*x+beta0)/beta_vec[1]
plt.scatter(x1[y==1],x2[y==1],marker='+')
plt.scatter(x1[y==-1],x2[y==-1],marker='s')
plt.plot(x,yy)
plt.show()