---
layout: post
title: Spectral Clustering
---

This blog post provides a tutorial on a simple version of the **spectral clustering** algorithm for clustering data points. In multivariate statistics, spectral clustering techniques make use of the spectrum (eigenvalues) of the similarity matrix of the data to perform dimensionality reduction before clustering in fewer dimensions. 

### Notation (inherited from the sample jupyter notebook)

In all the math below: 

- Boldface capital letters like $$\mathbf{A}$$ refer to matrices (2d arrays of numbers). 
- Boldface lowercase letters like $$\mathbf{v}$$ refer to vectors (1d arrays of numbers). 
- $$\mathbf{A}\mathbf{B}$$ refers to a matrix-matrix product (`A@B`). $$\mathbf{A}\mathbf{v}$$ refers to a matrix-vector product (`A@v`). 

**Cluster analysis** or **clustering** is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters).

## k-means Clustering

**k-means clustering** is a method of vector quantization that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. k-means works well on the following sample dataset (an example where we don't need spectral clustering).


```python
#import all the necessary packages to be used later
import scipy
import sklearn
import numpy as np
from sklearn import metrics
from sklearn import datasets
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
```


```python
n = 200
np.random.seed(1111)
X, y = datasets.make_blobs(n_samples=n, shuffle=True, random_state=None, centers = 2, cluster_std = 2.0)
plt.scatter(X[:,0], X[:,1])

plt.savefig("image-100.png") 
```



![image-100.png](/images/image-100.png)
    



```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(X)

plt.scatter(X[:,0], X[:,1], c = km.predict(X))

plt.savefig("image-101.png") 
```


    
![image-101.png](/images/image-101.png)
    


## Harder Clustering

k-means clustering, however, does not work well with all kind of datasets. See the following example where the Euclidean coordinates of the data points are contained in the matrix `X`, while the labels of each point are contained in `y`.


```python
np.random.seed(1234)
n = 200
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05, random_state=None)
plt.scatter(X[:,0], X[:,1])

plt.savefig("image-102.png") 
```


    
![image-102.png](/images/image-102.png)
    


Since k-means is designed to look for circular clusters, we cannot use it to cluster this sample dataset as we intended. 


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))

plt.savefig("image-103.png") 
```



![image-103.png](/images/image-103.png)
    


The above example provides us with motivation to use spectral clustering to cluster the two crescents.

## Spectral Clustering

### Part A: Construct the Similarity Matrix A

$$\mathbf{A}$$: the similarity matrix $$\mathbf{A}$$ (2d `np.ndarray`) with `n` rows and `n` columns and whose diagonal entries are all zeros

`n`: the number of data points

`epsilon`: the distance parameter (`A[i,j]` is `1` if `X[i]` is within `epsilon` of `X[j]`, and `0` otherwise)



```python
# for demonstration purpose, set epsilon to a specific value
epsilon = 0.4

# compute the distance matrix between each pair of data points 
A = euclidean_distances(X)

# compare each entry in the distance matrix with epsilon 
# assign entries less than epsilon to be 1, and others to be 0
A = (A < epsilon)*1

# set the diagonal entries to be 0
np.fill_diagonal(A,0)

# check our similarity matrix A
A
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 1, 0],
           ...,
           [0, 0, 0, ..., 0, 1, 1],
           [0, 0, 1, ..., 1, 0, 1],
           [0, 0, 0, ..., 1, 1, 0]])



### Part B: Calculate the Binary Norm Cut Objective of Matrix A

Now since matrix $$\mathbf{A}$$ contains distance information of the data points, the problem of clustering the data points is transferred to the problem of partitioning the rows and columns of matrix $$\mathbf{A}$$.

The **binary norm cut objective** of a matrix $$\mathbf{A}$$ is the function 

$$N_{\mathbf{A}}(C_0, C_1)\equiv \mathbf{cut}(C_0, C_1)\left(\frac{1}{\mathbf{vol}(C_0)} + \frac{1}{\mathbf{vol}(C_1)}\right)\;$$

In this above equation, 
- $$C_0$$, $$C_1$$: two clusters of the data points (a data point is either in $$C_0$$ or $$C_1$$)
- `y`: the label that specifies the cluster membership of each data point (e.g. `y[i] = 1` indicates point $${i \in C_0}$$, i.e. $i$th row of $$\mathbf{A}$$ belongs to $$C_1$$)
- $$\mathbf{cut}(C_0, C_1) \equiv \sum_{i \in C_0, j \in C_1} a_{ij}$$: the *cut* of the clusters $$C_0$$ and $$C_1$$
- $$\mathbf{vol}(C_0) \equiv \sum_{i \in C_0}d_i$$, where $$d_i = \sum_{j = 1}^n a_{ij}$$: the $$i$$th row sum of $$A$$



#### B.1 Compute The Cut Term

The cut term $$\mathbf{cut}(C_0, C_1)$$ is the sum of the entries `A[i,j]` where `i` and `j` belong to different clusters.


```python
def cut(A,y):
    # initilize the cut term to be 0
    cut = 0
    # use two for-loops to access y[i] and y[j]
    for i in range(n):
        for j in range(n):
            if y[i] == 0 and y[j] ==1:
                # add A[i,j] to the cut term if point i belongs to C0 and point j belongs to C1
                # do not need to consider the opposite scenario where point i belongs to C1 and point j belongs to C0
                # because the two scenarios are symmetrical and will cause duplication
                cut = cut + A[i,j]
    return cut
```

Apply the above function to compute the cut objective for the true clusters y and compare the cut objective for `y` with the cut objective of a random label.


```python
# generate a random vector of random labels of length n, with each label equal to either 0 or 1
rand = np.random.randint(0,2, size = n)
cut_obj_true_labels = cut(A,y)
cut_obj_random_labels = cut(A,rand)
print(cut_obj_true_labels)
print(cut_obj_random_labels)
```

    13
    1150


`cut_obj_true_labels` is much smaller than `cut_obj_random_labels` $=>$ the cut objective favors the true clusters over the random ones

#### B.2 Compute The Volume Term 

The volume term $$\mathbf{vol}(C_0)$$ measures the size of a of cluster.


```python
def vols(A,y):
    # v0 holds the volume of cluster 0 by summing up all rows who has label 0
    v0 = A[y==0,:].sum()
    # v1 holds the volume of cluster 1 by summing up all rows who has label 1
    v1 = A[y==1,:].sum()
    return v0,v1
```

#### B.3 Compute The Binary Norm Cut Objective 

$$N_{\mathbf{A}}(C_0, C_1)\equiv \mathbf{cut}(C_0, C_1)\left(\frac{1}{\mathbf{vol}(C_0)} + \frac{1}{\mathbf{vol}(C_1)}\right)\;$$


```python
def normcut(A,y):
    # cut(C0,C1) is computed by function cut(A,y)
    # vol(C0),vol(C1) are the first and second return values of vols(A,y) respectively
    return cut(A,y) * (1/(vols(A,y)[0]) + 1/(vols(A,y)[1]))
```

Compare the `normcut` objective with true labels `y` and the fake labels generated above. It is obvious that the binary norm cut objective for the true labels is much smaller.


```python
normcut_true = normcut(A,y)
normcut_rand = normcut(A, rand)
print(normcut_true)
print(normcut_rand)
```

    0.011518412331615225
    1.0240023597759158


### Part C: Math Trick

Now we aim to show that 


$$\mathbf{N}_{\mathbf{A}}(C_0, C_1) = \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}}\;$$

and also check the identity $$\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$

In the above expression,
- $$\mathbf{D}$$: the diagonal matrix with nonzero entries $$d_{ii} = d_i$$
- $$\mathbb{1}$$: the vector of `n` ones (i.e. `np.ones(n)`)
- $$\mathbf{z}$$: a new vector $$\mathbf{z} \in \mathbb{R}^n$$ such that: 

$$
z_i = 
\begin{cases}
    \frac{1}{\mathbf{vol}(C_0)} &\quad \text{if } y_i = 0 \\ 
    -\frac{1}{\mathbf{vol}(C_1)} &\quad \text{if } y_i = 1 \\ 
\end{cases}
$$

#### C.1 Compute **z**


```python
def transform(A,y):
    # assign z[i] to be 1/v0 when y[i] = 0
    # and assign z[i] to be -1/v1 when y[i] = 1
    z = 1/(vols(A,y)[0]) * (y == 0) - 1/(vols(A,y)[1]) * (y == 1)
    return z
```

#### C.2 Check The Equation


```python
# Compute the value of LHS:
normcut_true = normcut(A,y)
normcut_true
```




    0.011518412331615225




```python
# Compute the value of RHS:
z = transform(A,y)
D = np.zeros((n,n))
np.fill_diagonal(D,A.sum(axis=1))
matrix_product = ((z.T) @ (D - A) @ z )/ ((z.T) @ D @ z )
matrix_product
```




    0.011518412331615099




```python
# Check if both sides equal up to computer's precision
np.isclose(normcut_true,matrix_product)
```




    True



#### C.3 Check The Identity


```python
# Check if the left hand side is equal to 0 up to computer's precision
np.isclose(((z.T) @ D) @ np.ones(n),0)
```




    True



### Part D: Continuous Relaxation of The Normcut Problem

Part C shows that the problem of minimizing the normcut objective can be transferred to the problem of minimizing the function 

$$ R_\mathbf{A}(\mathbf{z})\equiv \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}} $$

subject to the condition $$\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$.

We can handle the condition by substituting for $$\mathbf{z}$$ the orthogonal complement of $$\mathbf{z}$$ relative to $$\mathbf{D}\mathbf{1}$$.


```python
def orth(u, v):
    return (u @ v) / (v @ v) * v

e = np.ones(n) 

d = D @ e

def orth_obj(z):
    z_o = z - orth(z, d)
    return (z_o @ (D - A) @ z_o)/(z_o @ D @ z_o)
```

Use the function `scipy.optimize.minimize` to minimize the function `orth_obj` with respect to $$\mathbf{z}$$.


```python
z_ = scipy.optimize.minimize(orth_obj,z,method='nelder-mead')
```


```python
z_min = z_.x
```

### Part E: Cluster the Data


```python
#get a copy of z_min
copy = np.copy(z_min)

#set the data points that are negative to have label 0
copy[copy < 0] = 0

#set the data points that are nonnegative to have label 1
copy[copy >= 0] = 1

#plot the data points with new labels
#data points with different labels are assigned different colors
plt.scatter(X[:,0], X[:,1], c = (z_min<0).astype(int)+1)

plt.savefig("image-104.png") 
```


    
![image-104.png](/images/image-104.png)
    


### Part F: 

The problem in Part E can actually be solved by using eigenvalues and eigenvectors of matrices.

Our objective is to minimize the function 

$$ R_\mathbf{A}(\mathbf{z})\equiv \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}} $$

with respect to $$\mathbf{z}$$, subject to the condition $$\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$.

This is equivalent to find the smallest eigenvalue of the generalized eigenvalue problem below according to the Rayleigh-Ritz Theorem.

$$ (\mathbf{D} - \mathbf{A}) \mathbf{z} = \lambda \mathbf{D}\mathbf{z}\;, \quad \mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$

This generalized eigenvalue problem is, again, equivalent to the standard eigenvalue problem 

$$ \mathbf{D}^{-1}(\mathbf{D} - \mathbf{A}) \mathbf{z} = \lambda \mathbf{z}\;, \quad \mathbf{z}^T\mathbb{1} = 0\;$$

And since $$\mathbb{1}$$ is the eigenvector with smallest eigenvalue of the matrix $$\mathbf{D}^{-1}(\mathbf{D} - \mathbf{A})$$, the vector $$\mathbf{z}$$ that minimizes the above function must be the eigenvector with the second-smallest eigenvalue. 




#### F.1 Construct the (normalized) Laplacian matrix L of the similarity matrix A

$$\mathbf{L} = \mathbf{D}^{-1}(\mathbf{D} - \mathbf{A})$$


```python
# find the inverse of D
D_inv = np.linalg.inv(D)
# construct L according to the formula
L = D_inv @ (D - A)
```

Find the eigenvector of L corresponding to its second-smallest eigenvalue.


```python
# compute all the eigenvalues and eigenvectors of L
Lam, U = np.linalg.eig(L)

# sort the eigenvalues and eigenvectors in increasing order
ix = Lam.argsort()
Lam, U = Lam[ix], U[:,ix]
# find the eigenvector with the second-smallest eigenvalue
z_eig = U[:,1]
```

Plot the data


```python
#get a copy of z_eig
copy = np.copy(z_eig)

#use the sign of z_eig as the color
#set the data points that are negative to have label 0
copy[copy < 0] = 0

#set the data points that are nonnegative to have label 1
copy[copy >= 0] = 1

#plot the data points with new labels
#data points with different labels are assigned different colors
plt.scatter(X[:,0], X[:,1], c = (z_eig<0).astype(int)+1)

plt.savefig("image-105.png") 
```


    
![image-105.png](/images/image-105.png)
    


### Part G: Synthesize Results


```python
def spectral_clustering(X, epsilon):
    """
    This function is used to perform spectral clustering
    It takes in
    X: the input data with n data points
    epsilon: the distance threshold that determines the label
    
    and returns an array of binary labels indicating whether data point i is in group 0 or group 1
    
    The assumption that ensures this function performs well is that the data points are in somewhat circular shape.
    """
    #construct the similarity matrix
    A = (euclidean_distances(X, X) < epsilon)*1
    np.fill_diagonal(A,0)
    
    #construct the Laplacian matrix
    D = np.zeros((n,n))
    np.fill_diagonal(D,A.sum(axis=1))
    L = np.linalg.inv(D) @ (D - A)
    
    #compute the eigenvector with second-smallest eigenvalue of the Laplacian matrix
    Lam, U = np.linalg.eig(L)
    U = U[:,Lam.argsort()]
    z_eig = U[:,1]
    
    #return labels based on this eigenvector
    return (z_eig<0).astype(int) + 1
```

Demonstration with the supplied original data from above.


```python
labels = spectral_clustering(X, epsilon)
```


```python
plt.scatter(X[:,0],X[:,1], c = labels)
plt.savefig("image-106.png") 
```


    
![image-106.png](/images/image-106.png)
    


### Part H: Other Datasets - make_moon

Now we can test our spectral clustering function with different datasets.
We can experiment with different noise level and we can also increase the number of data points.


```python
np.random.seed(1234)
n = 1000
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.03, random_state=None)
plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, epsilon))
plt.savefig("image-107.png") 
```


    
![image-107.png](/images/image-107.png)
    



```python
np.random.seed(1234)
n = 1000
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.08, random_state=None)
plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, epsilon))
plt.savefig("image-108.png") 
```


    
![image-108.png](/images/image-108.png)
    



```python
np.random.seed(1234)
n = 1000
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.12, random_state=None)
plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, epsilon))
plt.savefig("image-109.png") 
```


    
![image-109.png](/images/image-109.png)
    


As we can see from the plot above, our spectral clustering function still manages to find the two half-moon clusters. However, as the niose level increases, the number of points mis-clustered also start to increase and our function does not perform as well as at low noise levels.

### Part I: Other Datasets - the bull's eye


```python
n = 1000
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
plt.scatter(X[:,0], X[:,1])

plt.savefig("image-110.png") 
```


    
![image-110.png](/images/image-110.png)
    


We can see from below that k-means is unable to separate the data points to two circles.


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))

plt.savefig("image-111.png") 
```


    
![image-110.png](/images/image-111.png)
    


In comparison, from the experiments below, we find that our spectral clustering function correctly separates the two circles with epsilon values ranges roughly between 0.2 to 0.6.


```python
n = 1000
epsilon = 0.2
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
plt.scatter(X[:,0], X[:,1], c=spectral_clustering(X, epsilon))

plt.savefig("image-112.png") 
```


    
![image-110.png](/images/image-112.png)
    



```python
n = 1000
epsilon = 0.5
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
plt.scatter(X[:,0], X[:,1], c=spectral_clustering(X, epsilon))

plt.savefig("image-113.png") 
```


    
![image-110.png](/images/image-113.png)
    



```python
n = 1000
epsilon = 0.6
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
plt.scatter(X[:,0], X[:,1], c=spectral_clustering(X, epsilon))

plt.savefig("image-114.png") 
```


    
![image-110.png](/images/image-114.png)
    



```python
n = 1000
epsilon = 0.9
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
plt.scatter(X[:,0], X[:,1], c=spectral_clustering(X, epsilon))

plt.savefig("image-115.png") 
```


    
![image-110.png](/images/image-115.png)
    

