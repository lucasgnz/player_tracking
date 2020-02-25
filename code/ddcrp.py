#a python implementation of ddcrp according to the R version of Blei and Franzier
#written by placebo
#I only implement the framework with multinomial features, windows delay and linear distance function. you can implementation your own likelihood ,distance and delay function.
#have fun with it, more details of the algorithm can be found in Blei and Franzier's paper "Distance Dependent Chinese Restaurant Processes"

import numpy as np
from numpy import seterr, isneginf, array
from scipy.special import gammaln
from scipy.misc import logsumexp

def dirichlet_likelihood(Xp, hyper):
    if len(Xp.shape) == 2:
        X =sum(Xp)
    else:
        X = Xp
    idx = np.where(X!=0)
    lh = gammaln(len(X)*hyper) + sum(gammaln(X[idx]+hyper))\
    -len(idx)*gammaln(hyper)  - gammaln(sum(X)+len(X) * hyper)
    return lh

def linear_distance(i,j):
    return i-j

def window_delay(a,size=1):
    if abs(a) <= size and a >= 0:
        return 1;
    else:
        return 0;

#get the customers linked to customer i
def get_linked(i,link):
    c = []
    q = []
    q.append(i)
    while q:
        cur = q[0]
        c.append(cur)
        for k in range(0,len(link)):
            if (link[k] == cur) and (k not in c) and (k not in q):
                q.append(k)
        q = q[1:]
    return c

def ddcrp_infer(obs,lhood_fn,distance,delay,n_iter,alpha = 0.2):
    n = len(obs)
    cluster = np.array([0]*n)
    link = np.array([0]*n)
    prior = np.random.random(n*n).reshape((n,n))
    #lhood = np.random.random(n)
    merged_lhood = np.random.random(n)
    lhood = list(map(lambda x: lhood_fn(obs[np.where(cluster == x)]) , cluster))  #lhood of each cluster

    obs_lhood = 0 #the likelihood of all obs

    #prior of each customer
    for i in range(0,n):
        for j in range(0,n):
            try:
                if i==j:
                    prior[i][j] = np.log(alpha)
                else:
                    seterr(divide='ignore')
                    prior[i][j] = np.log(delay(distance(i,j)))
                    seterr(divide='warn')
                    prior[i][j][isneginf(prior[i][j])] = 0
            except Exception as e:
                # print(e)
                pass


    for t in range(0,n_iter):
        print("iter "+str(t))
        obs_lhood = 0
        for i in range(0,n):
            #print "sample"+str(i)+"th:"
            #remove the ith's link
            old_link = link[i]
            old_cluster = cluster[old_link]
            cluster[i] = i
            link[i] = i
            linked = get_linked(i,link)
            # print(linked)
            cluster[linked] = i

            if old_cluster not in linked :
                idx = np.where(cluster == old_cluster)
                lhood[old_cluster] = lhood_fn(obs[idx])
                lhood[i] = lhood_fn(obs[linked])


            #calculate the likelihood of the merged cluster
            for j in np.unique(cluster):

                if j == cluster[i] :
                    merged_lhood[j] = 2*lhood_fn(obs[linked])
                else:
                    merged_lhood[j] = lhood_fn(np.concatenate((obs[linked] , obs[np.where(cluster == j)])))

            log_prob = list(map(lambda x: prior[i][x] + merged_lhood[cluster[x]] - lhood[cluster[x]]-lhood[cluster[i]], np.arange(n)))
            prob = np.exp(log_prob - logsumexp(log_prob))

            #sample z_i
            link[i] = np.random.choice(np.arange(n),1,p=prob)

            #update the likelihood if the link sample merge two cluster
            new_cluster = cluster[link[i]]
            if new_cluster !=i:
                cluster[linked] = new_cluster
                lhood[new_cluster] = merged_lhood[new_cluster]

            #cal the likelihood of all obs
            for u in np.unique(cluster):
                obs_lhood = obs_lhood + lhood[u]

        print("cluster")
        print(cluster)
        print("link")
        print(link)
        print(obs_lhood)

    return cluster,link,obs_lhood



#a demo
obs = np.array([[10,0,0,0,0],[10,0,0,0,0],[5,0,0,0,0],[11,0,0,0,1],[0,10,0,0,1],[0,10,0,0,0],[0,0,10,0,0],[0,1,10,0,0],[20,0,2,0,0],[10,0,0,1,0],[10,1,0,10,0],[10,0,2,10,0],[10,0,0,10,0],[10,1,0,1,0],[10,0,0,0,0],])

#initalize parameters
n_iter =50
hyper = 0.01
alpha = 0.2
window_size = 20

#bookkeeper data
#cluster,link,obs,lhood,
lhood_fn = lambda x:dirichlet_likelihood(x,hyper)
distance = linear_distance
delay = lambda x:window_delay(x,window_size)

cluster,link,lhood = ddcrp_infer(obs,lhood_fn,distance,delay,n_iter,alpha)

print("-------")
print(cluster)
print(link)