#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.spatial.distance
import scipy

import csv


# In[2]:


GT = np.genfromtxt('ds0_Groundtruth.dat', skip_header = 3, skip_footer = 0, names = True, dtype = None, delimiter = ' ' , usecols = [0, 3, 5, 7])
Odom = np.genfromtxt('ds0_Odometry.dat', skip_header = 3, skip_footer = 0, names = True, dtype = None, delimiter = ' ' , usecols = [0,4,5])


# In[3]:


start_time = GT[0][0]
for i in range(len(GT)):
    GT[i][0]-= start_time
for i in range(len(Odom)):
    Odom[i][0]-=start_time


# In[4]:


j = 1
k = 1
align = []

for i in range (len(GT)):
    match = 0
    while (match == 0 and j < len(GT) and k < len(Odom)-1):
        if (GT[j][0] - Odom[k][0] > 0 and GT[j][0] - Odom[k+1][0]<= 0):
            align.append((GT[j], Odom[k]))
            j+=1
            match = 1
        else:
            k+=1


# In[5]:


train_in = []
train_out = []

test_in = []
test_out = []

val_in = []
val_out = []

ind = np.linspace(0, len(align), 10000)
ind_count = 0

for i in range(len(align) - 2):
    
    dt_in = align[i+1][0][0] - align[i][0][0]
    
    x_in = align[i+1][0][1] - align[i][0][1]
    y_in = align[i+1][0][2] - align[i][0][2]
    theta_in = align[i+1][0][3] - align[i][0][3]
    
    v_in = align[i+1][1][1] - align[i][1][1]
    w_in = align[i+1][1][2] - align[i][1][2]

    dt_out = align[i+2][0][0] - align[i+1][0][0] # (t+1) - (t)
    
    x_out = align[i+2][0][1] - align[i+1][0][1]
    y_out = align[i+2][0][2] - align[i+1][0][2]
    theta_out = align[i+2][0][3] - align[i+1][0][3]
    
    if (i == int(ind[ind_count]) and ind_count%2 ==0):
        test_in.append([x_in/dt_in, y_in/dt_in, theta_in/dt_in, v_in/dt_out, w_in/dt_out])
        test_out.append([x_out/dt_out, y_out/dt_out, theta_out/dt_out])
        ind_count+=1
    elif (i == int(ind[ind_count]) and ind_count%2 ==1):
        val_in.append([x_in/dt_in, y_in/dt_in, theta_in/dt_in, v_in/dt_out, w_in/dt_out])
        val_out.append([x_out/dt_out, y_out/dt_out, theta_out/dt_out])
        ind_count+=1
    else:
        train_in.append([x_in/dt_in, y_in/dt_in, theta_in/dt_in, v_in/dt_out, w_in/dt_out])
        train_out.append([x_out/dt_out, y_out/dt_out, theta_out/dt_out])
    


# In[6]:


train_in2 = []
train_out2 = []

test_in2 = []
test_out2 = []

val_in2 = []
val_out2 = []

ind = np.linspace(0, len(align), 10000)
ind_count = 0

for i in range(len(align) - 2):
    
    dt_in = align[i+1][0][0] - align[i][0][0]
    
    x_in = align[i+1][0][1] - align[i][0][1]
    y_in = align[i+1][0][2] - align[i][0][2]
    theta_in = align[i+1][0][3] - align[i][0][3]
    
    v = np.sqrt(((x_in/dt_in)**2)+((y_in/dt_in)**2))
    w = theta_in/dt_in
    
    v_in = align[i+1][1][1] - align[i][1][1]
    w_in = align[i+1][1][2] - align[i][1][2]
    
    del_v = np.abs(v - v_in)
    del_w = np.abs(w - w_in)

    dt_out = align[i+2][0][0] - align[i+1][0][0] # (t+1) - (t)
    
    x_out = align[i+2][0][1] - align[i+1][0][1]
    y_out = align[i+2][0][2] - align[i+1][0][2]
    theta_out = align[i+2][0][3] - align[i+1][0][3]
    
    if (i == int(ind[ind_count]) and ind_count%2 ==0):
        test_in2.append([del_v, del_w, dt_out])
        test_out2.append([x_out, y_out, theta_out])
        ind_count+=1
    elif (i == int(ind[ind_count]) and ind_count%2 ==1):
        val_in2.append([del_v, del_w, dt_out])
        val_out2.append([x_out, y_out, theta_out])
        ind_count+=1
    else:
        train_in2.append([del_v, del_w, dt_out])
        train_out2.append([x_out, y_out, theta_out])
    


# In[29]:


learning_set = []
for i in align:
    learning_set.append([i[0][0], i[0][1], i[0][2], i[0][3], i[1][1], i[1][1]])


# In[32]:


header = ['time, x, y, theta, v, w']

with open('learning_dataset.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)
    
    for data in learning_set:   
        # write the data
        writer.writerow(data)


# In[21]:


header = ['commanded v - robot instantaneous v', 'commanded w - robot instantaneous w', 'dt']

with open('test_input2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)
    
    for data in test_in2:   
        # write the data
        writer.writerow(data)


# In[7]:


def RBF_1d(X1, X2, l, var):
    '''
        l = lengthscale
        var = variance (sigma^2)
        
        scipy.spatial.distance.cdist(XA, XB, metric='euclidean', *, out=None, **kwargs)
        Compute distance between each pair of the two collections of inputs.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        
        Kxx = | k(x1,x1)   k(x1,x2)   ....   k(x1,xn) |
              |                                       |
              | k(xn,x1)   k(xn,x2)   ....   k(xn,xn) |
    '''
    K = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            d = np.abs(X1[i] - X2[j])
            K[i][j] = var*np.exp(-(d**2)/(2*(l**2)))
            #K[j][i] = var*np.exp(-(d**2)/(2*(l**2)))
    return K


# In[8]:


def GP_1d(x_in, y_out, x_test, l, var):
    '''
        x_in = 1D array of training input
        y_out = 1D array of training output
        x_test = single point of test input
    '''
    # compute kernel, covar of training input
    Kxx = RBF_1d(x_in, x_in, l, var)

    L = np.linalg.cholesky(Kxx + 1e-5*np.eye(x_in.shape[0])) # add noise
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
    
    alpha = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, y_out, lower=True)) # solve(a, b) --> ax = b --> x = a\b = a^-1b
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    
    # compute covar
    Kxxt = RBF_1d(x_in, x_test, l, var)

    # compute covar of test data
    Kxtxt = RBF_1d(x_test, x_test, l, var)
    
    # compute posterior mean
    mu_post = np.matmul(Kxxt.T, alpha)    
    
    #compute poserior covar
    v = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, Kxxt, lower=True))
    sigma_post = Kxtxt - np.matmul(Kxxt.T, v)
    
    #sigma = Kxtxt - np.dot(Kxxt, np.linalg.inv(Kxx).dot(Kxxt))
    # https://blog.dominodatalab.com/fitting-gaussian-process-models-python
    
    #mu_post = np.dot((scipy.linalg.solve(Kxx, Kxxt, assume_a='pos').T), y_out)
    #mu_post = (scipy.linalg.solve(Kxx, Kxxt, assume_a='pos').T) * y_out
    #sigma_post = Kxtxt - np.dot((scipy.linalg.solve(Kxx, Kxxt, assume_a='pos').T), Kxxt) 
    # https://peterroelants.github.io/posts/gaussian-process-tutorial/
    
    return mu_post, sigma_post


# In[92]:


x = np.random.uniform(-2*np.pi,2*np.pi,(100,1))
y = np.sin(x)
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('x vs sin(x)')
plt.show()


# In[10]:


x_test = np.random.uniform(-2*np.pi,2*np.pi,(100,1))

mu_test, sigma_test = GP_1d(x, y, x_test, 10, 0.1)
#print(mu_test.shape)
#print(sigma_test.shape)


# In[11]:


test_out_1 = np.random.multivariate_normal(mu_test.reshape(len(x_test)), sigma_test, len(x_test))
#print(test_out.shape)


# In[93]:


for i in range(len(x_test)):
    plt.scatter(x_test,test_out_1[i,:], c="black", alpha=0.5)
    plt.scatter(x, y, c = 'red')
    
plt.xlabel('x_test')
plt.ylabel('y_predict')
plt.title('lengthscale = 10, var = 0.1')

plt.show()


# In[99]:


mu_test2, sigma_test2 = GP_1d(x, y, x_test, 5, 0.1)

test_out_2 = np.random.multivariate_normal(mu_test2.reshape(len(x_test)), sigma_test2, len(x_test))

for i in range(len(x_test)):
    plt.scatter(x_test,test_out_2[i,:], c="black", alpha=0.5)
    plt.scatter(x, y, c = 'red')
    
plt.xlabel('x_test')
plt.ylabel('y_predict')
plt.title('lengthscale = 5, var = 0.1')
plt.show()


# In[95]:


mu_test3, sigma_test3 = GP_1d(x, y, x_test, 1, 0.1)

#test_out3 = np.random.multivariate_normal(mu_test3.reshape(len(x_test)), sigma_test3, len(x_test))
test_out_3 = np.random.multivariate_normal(mu_test3.reshape(len(x_test)), sigma_test3)

for i in range(len(x_test)):
    #plt.scatter(x_test,test_out3[i,:], c="black", alpha=0.5)
    plt.scatter(x_test[i],test_out_3[i], c="black", alpha=0.5)
    plt.scatter(x, y, c = 'red')
    
plt.xlabel('x_test')
plt.ylabel('y_predict')
plt.title('lengthscale = 1, var = 0.1')

plt.show()


# In[88]:


def RBF(X1, X2, l, var):
    '''
        l = lengthscale
        var = variance (sigma^2)
        
        scipy.spatial.distance.cdist(XA, XB, metric='euclidean', *, out=None, **kwargs)
        Compute distance between each pair of the two collections of inputs.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        
        Kxx = | k(x1,x1)   k(x1,x2)   ....   k(x1,xn) |
              |                                       |
              | k(xn,x1)   k(xn,x2)   ....   k(xn,xn) |
    '''
    d = scipy.spatial.distance.cdist(X1, X2, 'cityblock')
    K = var * np.exp(-(d**2)/(2*(l**2)))
    return K
    
def GP(x_in, y_out, x_test, l, var, noise, prior_mean):
    prior_mean = 0
    noise = 1e-5
    
    Kxx = RBF(x_in, x_in, l, var)
    
    L = np.linalg.cholesky(Kxx + 1e-5*np.eye(x_in.shape[0])) # add noise
    # Ax = b         where x = A\b
    # LL.T = A
    # Ly = b forward substitution
    # L.Tx = y back substitution
    # x = L.T \(L\b) solution
    # computation of L considered numerically stable

    alpha = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, y_out, lower=True)) # solve(a, b) --> ax = b --> x = a\b = a^-1b

    Kxxt = RBF(x_in, x_test, l, var)

    Kxtxt = RBF(x_test, x_test, l, var)

    mu_post = np.matmul(Kxxt.T, alpha)

    v = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, Kxxt, lower=True))
    sigma_post = Kxtxt - np.matmul(Kxxt.T, v)
    
    return mu_post, sigma_post
 
def log_like(x_in, y_out, x_test, l, var):
    Kxx = RBF(x_in, x_in, l, var)
    
    L = np.linalg.cholesky(Kxx + 1e-5*np.eye(x_in.shape[0])) # add noise

    alpha = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, y_out, lower=True)) # solve(a, b) --> ax = b --> x = a\b = a^-1b

    #MLL = (-.5*train_out.T*np.inv(Kxx)*train_out) - (.5*np.log(Kxx)) - (n/2)*np.log(2*np.pi)
    
    #np.sum(np.log(np.abs(K_chol))) + np.sum(scipy.linalg.solve_triangular((y_out), K_chol)**2) - 

    temp = 0
    n = len(x_in) #num training points
    for i in range(len(L)):
        temp+=np.log(L[i][i])
    MLL = -.5*y_out.T*alpha - temp - (n/2)*np.log(2*np.pi)
    return MLL

def log_like_1d(x_in, y_out, x_test, l, var):
    Kxx = RBF_1d(x_in, x_in, l, var)
    
    L = np.linalg.cholesky(Kxx + 1e-5*np.eye(x_in.shape[0])) # add noise

    alpha = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, y_out, lower=True)) # solve(a, b) --> ax = b --> x = a\b = a^-1b

    #MLL = (-.5*train_out.T*np.inv(Kxx)*train_out) - (.5*np.log(Kxx)) - (n/2)*np.log(2*np.pi)
    
    #np.sum(np.log(np.abs(K_chol))) + np.sum(scipy.linalg.solve_triangular((y_out), K_chol)**2) - 

    temp = 0
    n = len(x_in) #num training points
    for i in range(len(L)):
        temp+=np.log(L[i][i])
    MLL = -.5*y_out.T*alpha - temp - (n/2)*np.log(2*np.pi)
    return MLL
    
def least_squares(y_pred, y_true):
    total = 0
    for i in range(len(y_pred)):
        total+=((y_pred[i] - y_true[i])**2)
    return total

#def grad_descent():
    
def gradient_descent(loss_func, alpha_choice, max_its, params):
    g_flat, unflatten, w = flatten_func(loss_func, params) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
        
        # evaluate the gradient, store current (unflattened) weights and cost function value
        cost_eval,grad_eval = gradient(w)
        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

        # take gradient descent step
        params = params - alpha*grad_eval
            
    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing 
    # the gradient at the final step we don't get the final cost function value 
    # via the Automatic Differentiatoor) 
    cost_history.append(g_flat(w))  
    return weight_history[-1],cost_history[-1]

#def optimize():
        
#def log_max():

def MSE(mean_out, y_true):
    total = 0
    for i in range(len(mean_out)):
        total+=((mean_out[i][0] - y_true[i])**2)
    return total

def neg_log_prob(mean_out, y_true, post_var):
    noise = 1e-5*np.eye(post_var.shape[0])
    pred_var = post_var + noise
    #print((2*np.pi*pred_var))
    return .5*np.log(2*np.pi*(np.abs(pred_var))) + ((y_true-mean_out)**2)/(2*pred_var)


# In[16]:


y_true = np.sin(x_test)


# In[96]:


plt.scatter(x, y, color = 'black')
plt.scatter(x_test, y_true, color = 'red')
# plt.show()


# In[98]:


print('Least Squares with length scale = 10: ', least_squares(test_out_1[0], y_true))
print('Least Squares with length scale = 5: ', least_squares(test_out_2[0], y_true))
print('Least Squares with length scale = 1: ', least_squares(test_out_3, y_true))
print('\n')
print('MSE with length scale = 10: ', least_squares(mu_test[0], y_true))
print('MSE with length scale = 5: ', least_squares(mu_test2[0], y_true))
print('MSE with length scale = 1: ', least_squares(mu_test3, y_true))


# In[132]:


mu_test4, sigma_test4 = GP_1d(x, y, x_test, 5, 0.01)

test_out_4 = np.random.multivariate_normal(mu_test2.reshape(len(x_test)), sigma_test4, len(x_test))

for i in range(len(x_test)):
    plt.scatter(x_test,test_out_4[i,:], c="black", alpha=0.5)
    plt.scatter(x, y, c = 'red')
print('Least Squares with length scale = 5: ', least_squares(test_out_4[0], y_true))

plt.xlabel('x_test (black)\n x_training (red)')
plt.ylabel('y_predict (black)\n y_training (red)')
plt.title('lengthscale = 5, var = 0.01')

plt.show()


# In[ ]:




