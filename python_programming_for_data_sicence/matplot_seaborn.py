import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# kategorik değişkeni sutun grafiği ile gostermek gerekiyor.
#   Countplot, bar
# Sayılsal değişkeni histogram veya boxplot ile  
#   göstermek gerekiyor. 

titanic = sns.load_dataset('titanic')
df = titanic.copy()
df.head()

df["sex"].value_counts()

df["sex"].value_counts().plot(kind="bar", color="green")
plt.show()


plt.hist(df["age"], bins=30)
plt.show()

plt.boxplot(df["fare"])
plt.show()

# matplotlib özellikleri 

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y, color="red")
plt.show()

plt.plot(x, y, "o", color="red")
plt.show()

a = np.array([2, 4, 6, 8, 10])
b = np.array([1, 3, 5, 7, 9])

plt.plot(a, b, color="red")
plt.show()

plt.plot(a, b, "o", color="red")
plt.show()

# Marker özellikleri 

y = np.array([13, 28, 11, 100])

plt.plot(y, marker="o")
plt.show()

plt.plot(y, marker="*")
plt.show()


# line özellikleri

y = np.array([13, 28, 11, 100])

plt.plot(y, linestyle="--")
plt.show()

# multiple lines

y = np.array([13, 28, 11, 10])
x = np.array([23, 28, 11, 100])

plt.plot (x)
plt.plot (y)
plt.show()

# labels 

x = np.array([13, 28, 11, 10])
y = np.array([23, 28, 11, 100])
plt.plot(x, y)
plt.title("Grafik Başlığı")
plt.xlabel("x ekseni")
plt.ylabel("y ekseni")
plt.grid(True)
plt.show()


#subplot özellikleri

x = np.array ([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array ([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.subplot(1, 2, 1)
plt.title("Grafik 1")
plt.plot(x, y, color="red")


x = np.array ([80, 85, 90, 5, 100, 105, 111, 1121, 120, 125])
y = np.array ([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.subplot(1, 2, 2)
plt.title("Grafik 2")
plt.plot(x, y, color="green")
plt.show()

# SEABORN

tips = sns.load_dataset("tips")
df = tips.copy()
df.head()

df["sex"].value_counts()

sns.countplot(x="sex", data=df)
plt.show()

#sayısal değişkenlerin dağılımı

sns.boxplot(x=df["total_bill"])
plt.show()