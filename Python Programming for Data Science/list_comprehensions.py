salaryies = [1000, 2000, 3000, 4000, 5000]

new_salaryies = []

for i in salaryies:
    if i > 3000:
        new_salaryies.append(i)
    else:
        new_salaryies.append(i * 2)
        
print(new_salaryies)



[new_salaryies.append(i) if i > 3000 else new_salaryies.append(i * 2) for i in salaryies]

[salary* 2 for salary in salaryies if salary <= 3000]

[salary* 2 if salary <= 3000 else salary*0  for salary in salaryies]

students = ["ali", "veli", "deli", "ayşe"]

students_not = ["ali", "deli"]

[i.lower() if i in students_not else i.upper() for i in students]


"""
 Bir veri setindeki değişken isimlerini değiştirmek istiyoruz.
 
 before : + ['total', 'speeding', 'alcohol', 'not_distracted', "no_previous',"ins_premium','ins_losses', 'abbrev']
 
 After : + ["TOTAL", "SPEEDING", "ALCOHOL", "NOT_DISTRACTED", "NO_PREVIOUS", "INS_PREMIUM", "INS_LOSSES", "ABBREV"]
"""

import seaborn as sns

car_crashes=  sns.load_dataset("car_crashes")
df = car_crashes.copy()

df.columns = [col.upper() for col in df.columns]

[col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]