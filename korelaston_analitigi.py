import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = r"C:\Users\Lenova\Desktop\MiuulYazKampı\data\data.csv"
breast_cancer = pd.read_csv(file_path)
df = breast_cancer.copy()
df = df.iloc[:, 1:]
df.head()

num_cols = [col for col in df.columns if df[col].dtypes in [int, float]]

corr = df[num_cols].corr()

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(
    corr, annot=True, fmt=".2f", annot_kws={"size": 10}, linewidths=0.5, cmap="Blues"
)
plt.show()

# Yüksek korelasyonlu degiskenlerin silinmesi

corr_matrix = df.corr().abs()

upper_triange_matrix = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
)

