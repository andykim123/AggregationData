#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 02:16:28 2018

@author: dohoonkim
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt 

#np.seterr(divide='ignore', invalid='ignore')
#Assume ground truth to be 2
k = 2
epsilon = 0.999
f = math.factorial
alpha = 1 - epsilon + np.divide(epsilon,k)
beta = np.divide(epsilon,k)

prior_theta1 = 0.3 #parameter for prior, can be ranged from 0 to 1
prior_theta21 = 0.4 #ground truth prior on image
prior_theta22 = 0.3
prior_theta2 = prior_theta21 + prior_theta22

def Worker_type1(worker):
    #Assume number of signal counts to be 50
    n = 50
    #randomly generate the number of signal theta1 counts from n/4 to n/2 to prevent it from being too little
    count_theta1 = random.randrange(int(n/4),int(n/2))
    count_theta2 = n - count_theta1
    count_theta2_1 = random.randrange(np.ceil(int(count_theta2/2)),count_theta2)
    count_theta2_2 = count_theta2 - count_theta2_1
    if np.divide(count_theta2,count_theta1) > np.divide(count_theta2_1,count_theta2_2):
        return count_theta1, count_theta2_1, count_theta2_2
    else:
       return Worker_type1(worker)
   
def Worker_type2(worker):
    n = 50
    count_theta1 = random.randrange(int(n/4),int(n/2))
    count_theta2 = n - count_theta1
    count_theta2_1 = random.randrange(np.ceil(int(count_theta2/2)),count_theta2)
    count_theta2_2 = count_theta2 - count_theta2_1
    if np.divide(count_theta2,count_theta1) < np.divide(count_theta2_1,count_theta2_2):
        return count_theta1, count_theta2_1, count_theta2_2
    else:
       return Worker_type2(worker)
   
def Worker_posterior(c_theta1,c_theta2,c_theta3):
    total_count = c_theta1+c_theta2+c_theta3
    c_theta21 = c_theta2
    c_theta22 = c_theta3
    c_theta2sum = c_theta21 + c_theta22
    theta1_eq = np.power(alpha,c_theta1)*np.power(beta,total_count-c_theta1)*prior_theta1
    theta2_eq = np.power(alpha,c_theta2sum)*np.power(beta,total_count-c_theta2sum)*prior_theta2
    theta1_eqn = np.divide(theta1_eq,theta1_eq+theta2_eq)
    theta2_eqn = np.divide(theta2_eq,theta1_eq+theta2_eq)
    theta21_eq = np.power(alpha,c_theta21)*np.power(beta,c_theta22)*prior_theta21
    theta22_eq = np.power(alpha,c_theta22)*np.power(beta,c_theta21)*prior_theta22
    theta21_eqn = np.divide(theta21_eq,theta21_eq+theta22_eq)
    theta22_eqn = np.divide(theta22_eq,theta22_eq+theta21_eq)
    posterior_theta1 = theta1_eqn
    posterior_theta21 = np.multiply(theta2_eqn,theta21_eqn)
    posterior_theta22 = np.multiply(theta2_eqn,theta22_eqn)
    return posterior_theta1,posterior_theta21,posterior_theta22

#calculate the signal difference counts for corresponding worker type
def signal_difference(post1,post21,post22):
    diff1 = np.divide(math.log(post21 + post22)-math.log(post1)-math.log(prior_theta2)+math.log(prior_theta1),math.log(alpha) - math.log(beta))
    diff2 = np.divide(math.log(post21)-math.log(post22)-math.log(prior_theta21)+math.log(prior_theta22),math.log(alpha) - math.log(beta))
    return diff1,diff2

def Q1_W1_posterior(total_sig_diff):
#    constant = 2000
    k = total_sig_diff
    c1 = 20 #set constant for count on theta1
    c2 = c1 + k
    c2_1 = int(c2/2)
    c2_2 = int(c2/2)
#    pl1_eq = np.power(alpha,c1)*np.power(beta,c2)*prior_theta1
#    pl2_eq = np.power(alpha,c2)*np.power(beta,c1)*prior_theta2
    logpl1 = np.log(prior_theta1) + c1*np.log(alpha) + c2*np.log(beta)
    logpl2 = np.log(prior_theta1) + c2*np.log(alpha) + c1*np.log(beta)
#    print(logpl1)
#    print(logpl2)
    number_dec = -int(str(logpl1).split('.')[0])
    constant = int(round(number_dec, -2))
    logpl1t = logpl1 + constant
    logpl2t = logpl2 + constant
    
    exppl1 = np.exp(logpl1t)
    exppl2 = np.exp(logpl2t)
#    print(exppl1)
#    print(exppl2)
    pl1_eqn = np.divide(exppl1,exppl1+exppl2)
#    print(pl1_eqn)
    pl2_eqn = np.divide(exppl2,exppl1+exppl2)
    
    logpl21 = np.log(prior_theta1) + c2_1*np.log(alpha) + c2_2*np.log(beta)
    logpl22 = np.log(prior_theta1) + c2_2*np.log(alpha) + c2_1*np.log(beta)

   
    logpl21t = logpl21 + constant
    logpl22t = logpl22 + constant
    exppl21 = np.exp(logpl21t)
    exppl22 = np.exp(logpl22t)
#    print(exppl21)
#    print(exppl22)
    pl21_eqn = np.divide(exppl21,exppl21+exppl22)
    pl22_eqn = np.divide(exppl22,exppl21+exppl22)
    
#    pl21_eq = np.power(alpha,c2_1)*np.power(beta,c2_2)*prior_theta21
#    pl22_eq = np.power(alpha,c2_2)*np.power(beta,c2_1)*prior_theta22
#    pl21_eqn = np.divide(pl21_eq,pl21_eq+pl22_eq)
#    pl22_eqn = np.divide(pl22_eq,pl22_eq+pl21_eq)
    plposterior_theta1 = pl1_eqn
    plposterior_theta21 = np.multiply(pl2_eqn,pl21_eqn)
    plposterior_theta22 = np.multiply(pl2_eqn,pl22_eqn)
    return plposterior_theta1,plposterior_theta21,plposterior_theta22

def Q2_W2_posterior(total_sig_diff1_2,total_sig_diff2):
    k = total_sig_diff2
    j = total_sig_diff1_2
#    constant = 3000
    c1 = 20
    c2_2 = int((c1+j-k)/2)
    c2_1 = int((c1+k+j)/2)
    c2 = c2_1 + c2_2
#    pl1_eq = np.power(alpha,c1)*np.power(beta,c2)*prior_theta1
#    pl2_eq = np.power(alpha,c2)*np.power(beta,c1)*prior_theta2
#    pl1_eqn = np.divide(pl1_eq,pl1_eq+pl2_eq)
#    pl2_eqn = np.divide(pl2_eq,pl1_eq+pl2_eq)
#    pl21_eq = np.power(alpha,c2_1)*np.power(beta,c2_2)*prior_theta21
#    pl22_eq = np.power(alpha,c2_2)*np.power(beta,c2_1)*prior_theta22
#    pl21_eqn = np.divide(pl21_eq,pl21_eq+pl22_eq)
#    pl22_eqn = np.divide(pl22_eq,pl22_eq+pl21_eq)
    logpl1 = np.log(prior_theta1) + c1*np.log(alpha) + c2*np.log(beta)
    logpl2 = np.log(prior_theta1) + c2*np.log(alpha) + c1*np.log(beta)
    number_dec = -int(str(logpl1).split('.')[0])
#    print(number_dec)
    constant = int(round(number_dec, -2))
#    print(constant)
    logpl1t = logpl1 + constant
    logpl2t = logpl2 + constant
    exppl1 = np.exp(logpl1t)
    exppl2 = np.exp(logpl2t)
#    print(exppl1)
#    print(exppl2)
    pl1_eqn = np.divide(exppl1,exppl1+exppl2)
#    print(pl1_eqn)
    pl2_eqn = np.divide(exppl2,exppl1+exppl2)
    
    logpl21 = np.log(prior_theta1) + c2_1*np.log(alpha) + c2_2*np.log(beta)
    logpl22 = np.log(prior_theta1) + c2_2*np.log(alpha) + c2_1*np.log(beta)
    logpl21t = logpl21 + constant
    logpl22t = logpl22 + constant
    exppl21 = np.exp(logpl21t)
    exppl22 = np.exp(logpl22t)
#    print(exppl21)
#    print(exppl22)
    pl21_eqn = np.divide(exppl21,exppl21+exppl22)
    pl22_eqn = np.divide(exppl22,exppl21+exppl22)
    plposterior_theta1 = pl1_eqn
    plposterior_theta21 = np.multiply(pl2_eqn,pl21_eqn)
    plposterior_theta22 = np.multiply(pl2_eqn,pl22_eqn)
    return plposterior_theta1,plposterior_theta21,plposterior_theta22

def Q1_W2_posterior(total_sig_diff):
    k = total_sig_diff
    c1 = 20
    c2 = c1 + k
    c2_1 = int(c2/2)
    c2_2 = int(c2/2)
#    pl1_eq = np.power(alpha,c1)*np.power(beta,c2)*prior_theta1
#    pl2_eq = np.power(alpha,c2)*np.power(beta,c1)*prior_theta2
#    pl1_eqn = np.divide(pl1_eq,pl1_eq+pl2_eq)
#    pl2_eqn = np.divide(pl2_eq,pl1_eq+pl2_eq)
#    pl21_eq = np.power(alpha,c2_1)*np.power(beta,c2_2)*prior_theta21
#    pl22_eq = np.power(alpha,c2_2)*np.power(beta,c2_1)*prior_theta22
#    pl21_eqn = np.divide(pl21_eq,pl21_eq+pl22_eq)
#    pl22_eqn = np.divide(pl22_eq,pl22_eq+pl21_eq)
    logpl1 = np.log(prior_theta1) + c1*np.log(alpha) + c2*np.log(beta)
    logpl2 = np.log(prior_theta1) + c2*np.log(alpha) + c1*np.log(beta)
    number_dec = -int(str(logpl1).split('.')[0])
#    print(number_dec)
    constant = int(round(number_dec, -2))
#    print(constant)
    logpl1t = logpl1 + constant
    logpl2t = logpl2 + constant
    exppl1 = np.exp(logpl1t)
    exppl2 = np.exp(logpl2t)
#    print(exppl1)
#    print(exppl2)
    pl1_eqn = np.divide(exppl1,exppl1+exppl2)
#    print(pl1_eqn)
    pl2_eqn = np.divide(exppl2,exppl1+exppl2)
    
    logpl21 = np.log(prior_theta1) + c2_1*np.log(alpha) + c2_2*np.log(beta)
    logpl22 = np.log(prior_theta1) + c2_2*np.log(alpha) + c2_1*np.log(beta)
    logpl21t = logpl21 + constant
    logpl22t = logpl22 + constant
    exppl21 = np.exp(logpl21t)
    exppl22 = np.exp(logpl22t)
#    print(exppl21)
#    print(exppl22)
    pl21_eqn = np.divide(exppl21,exppl21+exppl22)
    pl22_eqn = np.divide(exppl22,exppl21+exppl22)
    plposterior_theta1 = pl1_eqn
    plposterior_theta21 = np.multiply(pl2_eqn,pl21_eqn)
    plposterior_theta22 = np.multiply(pl2_eqn,pl22_eqn)
    return plposterior_theta1,plposterior_theta21,plposterior_theta22

def Q2_W1_posterior(total_sig_diff1_2,total_sig_diff):
    k = total_sig_diff
    j = total_sig_diff1_2
    print(k)
    print(j)
    c1 = 20
    c2_2 = int((c1+j-k)/2)
    c2_1 = int((c1+k+j)/2)
    c2 = c2_1 + c2_2
#    c1 = c2 + j
#    pl1_eq = np.power(alpha,c1)*np.power(beta,c2)*prior_theta1
#    pl2_eq = np.power(alpha,c2)*np.power(beta,c1)*prior_theta2
#    pl1_eqn = np.divide(pl1_eq,pl1_eq+pl2_eq)
#    pl2_eqn = np.divide(pl2_eq,pl1_eq+pl2_eq)
#    pl21_eq = np.power(alpha,c2_1)*np.power(beta,c2_2)*prior_theta21
#    pl22_eq = np.power(alpha,c2_2)*np.power(beta,c2_1)*prior_theta22
#    pl21_eqn = np.divide(pl21_eq,pl21_eq+pl22_eq)
#    pl22_eqn = np.divide(pl22_eq,pl22_eq+pl21_eq)
    logpl1 = np.log(prior_theta1) + c1*np.log(alpha) + c2*np.log(beta)
    logpl2 = np.log(prior_theta1) + c2*np.log(alpha) + c1*np.log(beta)
#    print(logpl1)
#    if logpl1 < 0:
#        number_dec = -int(str(logpl1).split('.')[0])
#        constant = int(round(number_dec, -1))
#    else:
#        number_dec = int(str(logpl1).split('.')[0])
#        constant = -int(round(number_dec, -1))
    number_dec = -int(str(logpl1).split('.')[0])
    constant = int(round(number_dec, -2))
#    print(number_dec)   
#    print(constant)

    logpl1t = logpl1 + constant
    logpl2t = logpl2 + constant
#    print(logpl1t)
#    print(logpl2)
    exppl1 = np.exp(logpl1t)
    exppl2 = np.exp(logpl2t)
#    print(exppl1)
#    print(exppl2)
    pl1_eqn = np.divide(exppl1,exppl1+exppl2)
#    print(pl1_eqn)
    pl2_eqn = np.divide(exppl2,exppl1+exppl2)
    
    logpl21 = np.log(prior_theta1) + c2_1*np.log(alpha) + c2_2*np.log(beta)
    logpl22 = np.log(prior_theta1) + c2_2*np.log(alpha) + c2_1*np.log(beta)
    logpl21t = logpl21 + constant
    logpl22t = logpl22 + constant
    exppl21 = np.exp(logpl21t)
    exppl22 = np.exp(logpl22t)
#    print(exppl21)
#    print(exppl22)
    pl21_eqn = np.divide(exppl21,exppl21+exppl22)
    pl22_eqn = np.divide(exppl22,exppl21+exppl22)
    plposterior_theta1 = pl1_eqn
    plposterior_theta21 = np.multiply(pl2_eqn,pl21_eqn)
    plposterior_theta22 = np.multiply(pl2_eqn,pl22_eqn)
    return plposterior_theta1,plposterior_theta21,plposterior_theta22

#generate 25 workers with type1 and calculate signal differences 
num_workers = range(1,501)
worker1_sigdiff1 = []
worker1_sigdiff2 = []
for i in range(500):
    sig1, sig2_1, sig2_2 = Worker_type1(i)
    worker_post1,worker_post2_1,worker_post2_2 = Worker_posterior(sig1,sig2_1,sig2_2)
    sig_diff1, sig_diff2 = signal_difference(worker_post1,worker_post2_1,worker_post2_2)
    worker1_sigdiff1.append(sig_diff1)
    worker1_sigdiff2.append(sig_diff2)

#generate 25 workers with type2 and calculate signal differences 
worker2_sigdiff1 = []
worker2_sigdiff2 = []
for i in range(500):
    sig1, sig2_1, sig2_2 = Worker_type2(i)
    worker_post1,worker_post2_1,worker_post2_2 = Worker_posterior(sig1,sig2_1,sig2_2)
    sig_diff1, sig_diff2 = signal_difference(worker_post1,worker_post2_1,worker_post2_2)
    worker2_sigdiff1.append(sig_diff1)
    worker2_sigdiff2.append(sig_diff2)    

#calculate platform posterior for Q1_Worker1 pair
total_w1_sigdiff1 = 0
Q1_W1_plpost1 = []
Q1_W1_plpost2_1 = []
Q1_W1_plpost2_2 = []
for sigdiff in worker1_sigdiff1:
    total_w1_sigdiff1 += sigdiff
    platform_post1, platform_post2_1, platform_post2_2 = Q1_W1_posterior(total_w1_sigdiff1)
    Q1_W1_plpost1.append(platform_post1)
    Q1_W1_plpost2_1.append(platform_post2_1)
    Q1_W1_plpost2_2.append(platform_post2_2)

#calculate platform posterior for Q2_Worker2 pair
total_w2_sigdiff2 = 0
total_w2_sigdiff1_2 = 0
Q2_W2_plpost1 = []
Q2_W2_plpost2_1 = []
Q2_W2_plpost2_2 = []
#for sigdiff in worker2_sigdiff2:
#    total_w2_sigdiff2 += sigdiff
#    platform_post1, platform_post2_1, platform_post2_2 = Q2_W2_posterior(total_w2_sigdiff2)
#    Q2_W2_plpost1.append(platform_post1)
#    Q2_W2_plpost2_1.append(platform_post2_1)
#    Q2_W2_plpost2_2.append(platform_post2_2)

for i in range(len(worker2_sigdiff2)):
    total_w2_sigdiff2 += worker2_sigdiff2[i]
    total_w2_sigdiff1_2 += worker2_sigdiff1[i]
    platform_post1, platform_post2_1, platform_post2_2 = Q2_W2_posterior(total_w2_sigdiff1_2,total_w2_sigdiff2)
    Q2_W2_plpost1.append(platform_post1)
    Q2_W2_plpost2_1.append(platform_post2_1)
    Q2_W2_plpost2_2.append(platform_post2_2)

#calculate platform posterior for Q2_Worker1 pair
total_w1_sigdiff2 = 0
total_w1_sigdiff1_2 = 0
Q2_W1_plpost1 = []
Q2_W1_plpost2_1 = []
Q2_W1_plpost2_2 = []
#for sigdiff in worker1_sigdiff2:
#    total_w1_sigdiff2 += sigdiff
#    platform_post1, platform_post2_1, platform_post2_2 = Q2_W1_posterior(total_w1_sigdiff2)
#    Q2_W1_plpost1.append(platform_post1)
#    Q2_W1_plpost2_1.append(platform_post2_1)
#    Q2_W1_plpost2_2.append(platform_post2_2)

for i in range(len(worker1_sigdiff2)):
    total_w1_sigdiff2 += worker1_sigdiff2[i]
    total_w1_sigdiff1_2 += worker1_sigdiff1[i]
    platform_post1, platform_post2_1, platform_post2_2 = Q2_W1_posterior(total_w1_sigdiff1_2,total_w1_sigdiff2)
    Q2_W1_plpost1.append(platform_post1)
    Q2_W1_plpost2_1.append(platform_post2_1)
    Q2_W1_plpost2_2.append(platform_post2_2)

#calculate platform posterior for Q1_Worker2 pair
total_w2_sigdiff1 = 0
Q1_W2_plpost1 = []
Q1_W2_plpost2_1 = []
Q1_W2_plpost2_2 = []
for sigdiff in worker2_sigdiff1:
    total_w2_sigdiff1 += sigdiff
    platform_post1, platform_post2_1, platform_post2_2 = Q1_W2_posterior(total_w2_sigdiff1)
    Q1_W2_plpost1.append(platform_post1)
    Q1_W2_plpost2_1.append(platform_post2_1)
    Q1_W2_plpost2_2.append(platform_post2_2)

#Information gain
entropy_Q1W1 = []
entropy_Q1W2 = []
entropy_Q2W1 = []
entropy_Q2W2 = []

#update entropy for Q1W1
for i in range(500):
    H = -(Q1_W1_plpost1[i]*np.log(Q1_W1_plpost1[i]) + Q1_W1_plpost2_1[i]*np.log(Q1_W1_plpost2_1[i]) + Q1_W1_plpost2_2[i]*np.log(Q1_W1_plpost2_2[i]))
    entropy_Q1W1.append(H)

for i in range(500):
    H = -(Q1_W2_plpost1[i]*np.log(Q1_W2_plpost1[i]) + Q1_W2_plpost2_1[i]*np.log(Q1_W2_plpost2_1[i]) + Q1_W2_plpost2_2[i]*np.log(Q1_W2_plpost2_2[i]))
    entropy_Q1W2.append(H)
    
for i in range(500):
    H = -(Q2_W1_plpost1[i]*np.log(Q2_W1_plpost1[i]) + Q2_W1_plpost2_1[i]*np.log(Q2_W1_plpost2_1[i]) + Q2_W1_plpost2_2[i]*np.log(Q2_W1_plpost2_2[i]))
    entropy_Q2W1.append(H)
    
for i in range(500  ):
    H = -(Q2_W2_plpost1[i]*np.log(Q2_W2_plpost1[i]) + Q2_W2_plpost2_1[i]*np.log(Q2_W2_plpost2_1[i]) + Q2_W2_plpost2_2[i]*np.log(Q2_W2_plpost2_2[i]))
    entropy_Q2W2.append(H)

print("Degree of minimization of Q1W1 entropy")
print(entropy_Q1W1[0] - entropy_Q1W1[-1])
print("Degree of minimization of Q2W1 entropy")
print(entropy_Q2W1[0] - entropy_Q2W1[-1])
print("Degree of minimization of Q2W2 entropy")
print(entropy_Q2W2[0] - entropy_Q2W2[-1])
print("Degree of minimization of Q1W2 entropy")
print(entropy_Q1W2[0] - entropy_Q1W2[-1])


print("minimized entropy of Q1W1 pair")
print(entropy_Q1W1[-1])
print("minimized entropy of Q2W1 pair")
print(entropy_Q2W1[-1])
print("minimized entropy of Q2W2 pair")
print(entropy_Q2W2[-1])
print("minimized entropy of Q1W2 pair")
print(entropy_Q1W2[-1])


#Plot graphs for accuracy
plt.figure()
plt.plot(num_workers, Q1_W1_plpost2_1) 
plt.xlabel('number of workers')
plt.ylabel('Q1_W1_accuracy')
plt.title('Q1_W1_accuracy vs number of workers')  

plt.figure()
plt.plot(num_workers, Q2_W1_plpost2_1) 
plt.xlabel('number of workers')
plt.ylabel('Q2_W1_accuracy')
plt.title('Q2_W1_accuracy vs number of workers') 

plt.figure()
plt.plot(num_workers, Q2_W2_plpost2_1) 
plt.xlabel('number of workers')
plt.ylabel('Q2_W2_accuracy')
plt.title('Q2_W2_accuracy vs number of workers')  

plt.figure()
plt.plot(num_workers, Q1_W2_plpost2_1) 
plt.xlabel('number of workers')
plt.ylabel('Q1_W2_accuracy')
plt.title('Q1_W2_accuracy vs number of workers')  

#Plot graphs for entropy
plt.figure()
plt.plot(num_workers, entropy_Q1W1) 
plt.xlabel('number of workers')
plt.ylabel('Q1_W1_entropy')
plt.title('Q1_W1_entropy vs number of workers')  

plt.figure()
plt.plot(num_workers, entropy_Q1W2) 
plt.xlabel('number of workers')
plt.ylabel('Q1_W2_entropy')
plt.title('Q1_W2_entropy vs number of workers')  

plt.figure()
plt.plot(num_workers, entropy_Q2W2) 
plt.xlabel('number of workers')
plt.ylabel('Q2_W2_entropy')
plt.title('Q2_W2_entropy vs number of workers')  

plt.figure()
plt.plot(num_workers, entropy_Q2W1) 
plt.xlabel('number of workers')
plt.ylabel('Q2_W1_entropy')
plt.title('Q2_W1_entropy vs number of workers')  

plt.show()

