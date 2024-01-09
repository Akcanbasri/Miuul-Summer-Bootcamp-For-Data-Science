import numpy as np

a = [1,2,3,4,5]
b = [6,7,8,9,10]

ab = []

for i in range(0,len(a)):
    ab.append(a[i]*b[i])


a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])

a*b

#numpy array oluşturma

np.array([1,2,3,4,5])

# sıfırdan array oluşturma
np.zeros(10 , dtype = int)

# random array oluşturma
np.random.randint(0,10, size = 10)

# normal dağılımlı array oluşturma
np.random.normal(10,4, size = (3,4))

# özellikleri
a = np.random.randint(10, size = 5)
a.ndim
a.shape
a.size
a.dtype

# reshape

a = np.random.randint(1, 9, size = 9)
a.reshape(3,3)

# indexing

a = np.random.randint(10, size = 10)
a[0]
a[0:3]
a[0] = 999

b = np.random.randint(10, size = (3,5))

b[0,0]
b[1, 1]
b[0, :]
b[: , 0]
b[0:2, 0:3]

# fancy indexing

v = np.arange(0,30,3)
v[1]
catch = [1, 2, 3]
v[catch]

# conditional indexing
v = np.array([1,2,3,4,5])

v[v<3]
v[v !=3]

# matematiksel işlemler
c = np.array([1,2,3,4,5])
c/5

c* 5 /10

c**2

np.subtract(c,1)
np.add(c,1)
np.multiply(c,2)
np.mean(c)
np.max(c)
np.min(c)
np.var(c)

# iki bilinmeyenli denklem çözümü

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5,1], [1,3]])
b = np.array([12,10])

np.linalg.solve(a,b)