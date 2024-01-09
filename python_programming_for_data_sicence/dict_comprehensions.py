dictionary = {
    "A":1,
    "B":2,
    "C":3,
    "D":4,
}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for k, v in dictionary.items()}
{k.lower(): v for k, v in dictionary.items()}
{k.lower(): v ** 2 for k, v in dictionary.items()}


numbers = range(10)
new_dict = {}


for i in numbers:
    if i % 2 == 0:
        new_dict[i] = i ** 2
        
print(new_dict)

{n: n**2 for n in numbers if n % 2 == 0} # mülakat sorusu


import seaborn as sns

car_crashes = sns.load_dataset("car_crashes")
df = car_crashes.copy()
df.columns

num_columns = [ col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list= ["mean", "max", "min", "sum"]

# long way
for col in num_columns:
    soz[col] = agg_list

# short way
new_dict = {col: agg_list for col in num_columns}

df[num_columns].head()
df[num_columns].agg(new_dict)


