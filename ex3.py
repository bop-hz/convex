import cvxpy as cp
import numpy as np
x= cp.Variable()
y=cp.Variable()
z = cp.Variable()
expr = cp.power(x,2)+cp.power(y,2)+cp.power(z,2) <=2
print((expr.is_dcp()))
expr2 = cp.norm(cp.hstack([1,x]),2)-3*x <=y
print((expr2.is_dcp()))
expr3 = [cp.power(x,-1)+cp.power(y,-1) <=5,x>=0,y>=0]
prob = cp.Problem(cp.Minimize(1),expr3)
print(prob.is_dcp())
expr4 = [x+2*y==0,x-y==0]
prob1 = cp.Problem(cp.Minimize(1),expr4)
print(prob1.is_dcp())
u = cp.Variable()
v = cp.Variable()
s = cp.Variable()
expr5 = [cp.quad_over_lin(cp.power(u,2),v)<=s,s==y,u==x+y,v==x-y+5]
prob2 = cp.Problem(cp.Minimize(1),expr5)
print(prob2.is_dcp())
expr6 = [-cp.log(u)-cp.log(v)<=0,u==x+z]
prob3 = cp.Problem(cp.Minimize(1),expr6)
print(prob3.is_dcp())
expr7 = -cp.log(x)-0.5*cp.log(y)<=0
print(expr7.is_dcp())
expr8 = [cp.log_sum_exp(cp.hstack([u,v]))+cp.exp(x)<=0,u==y-1,v==0.5*x]
prob4 = cp.Problem(cp.Minimize(1),expr8)
print(prob4.is_dcp())