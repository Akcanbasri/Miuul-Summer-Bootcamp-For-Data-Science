"""
Görev 1:  Verilen değerlerin veri yapılarını inceleyiniz.Type() metodunu kullanınız.

x = 8 -- int 
y = 3.2 -- float
z = 8 + 18 -- int
a = "Hello World" -- str
b = True -- bool
c = 23 < 22 -- bool
l = [1, 2, 3, 4] -- list
d = {"Name": "Jake", 
     "Age": 27, 
     "Adress": "Downtown" } -- dict
t = ("Machine Learning", "Data Science") -- tuple
s = {"Python", "Machine Learning", "Data Science"} -- set
"""

x = 8 
type(x)
y = 3.2
type(y)
z = 8 + 18
type(z)
a = "Hello World"
type(a)
b = True
type(b)
c = 23 < 22
type(c)
l = [1, 2, 3, 4]
type(l)
d = {"Name": "Jake",
     "Age": 27, 
     "Adress": "Downtown" }
type(d)
t = ("Machine Learning", "Data Science")
type(t)
s = {"Python", "Machine Learning", "Data Science"}
type(s)


"""
    Görev 2:  Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
    
    text = "The goal is to turn data into information, and information into insight."
    
    beklenen = ['THE', 'GOAL', 'IS', 'TO', 'TURN', 'DATA', 'INTO', 'INFORMATION', 'AND', 'INFORMATION', 'INTO', 'INSIGHT']
"""

text = "The goal is to turn data into information, and information into insight."

text.upper().replace(",", "").replace(".", "").split()

""" 
Görev 3:  Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
 
Adım1: Verilen listeni eleman sayısına bakınız.
Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesioluşturunuz.
Adım4: Sekizinci indeksteki elemanı siliniz.
Adım5: Yeni bir eleman ekleyiniz.
Adım6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

"""

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım1: Verilen listeninelemansayısınabakınız.
len(lst)

# Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
lst[0]
lst[10]

# Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesioluşturunuz.
new_list = lst[0:4]

# Adım4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)

# Adım5: Yeni bir eleman ekleyiniz.
lst.append("X")

# Adım6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")

""" 
Görev 4:  Verilen sözlük yapısına aşağıdaki adımları uygulayınız

dict = {'Christian': ["America", 18],
'Daisy': ["England", 12],
'Antonio': ["Spain", 22],
'Dante': ["Italy" ,25]}

Adım1: Key değerlerine erişiniz.
Adım2: Value'lara erişiniz.
Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
Adım5: Antonio'yu dictionary'den siliniz.

"""

dict = {'Christian': ["America", 18],
'Daisy': ["England", 12],
'Antonio': ["Spain", 22],
'Dante': ["Italy" ,25]}

# Adım1: Key değerlerine erişiniz.
dict.keys()
# Adım2: Value'lara erişiniz.
dict.values()
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = 13
# Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict["Ahmet"] = ["Turkey", 24]
# Adım5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")


""" 
Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayilari ayr listelere atayan ve bu liste return eden fonksiyon yaziniz.
"""

lst = [2,13, 18,93,22]

def odd_even(lst):
    odd = []
    even = []
    for i in lst:
        if i % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    return odd, even

even_list, odd_list = odd_even(lst)


"""
Görev 6: Asagida verilen listede mühendislik ve tip fakülterinde dereceye giren ögrencilerin isimleri bulunmaktadir. Sirasiyla ilk üg ögrenci mühendislik 
fakültesinin basari sirasini temsil ederken son üç ögrenci de tip fakültesi ögrenci sirasina aittir. Enumarate kullanarak ögrenci derecelerini 
fakülte özelinde yazdirinIz.

ogrenciler = ["Ali", "Veli", "Ayse", "Talat", "Zeynep", "Ece"]

Beklenen Çıktı:
Mühendislik Fakültesi 1 . ögrenci: Ali
Mühendislik Fakültesi 2 . ögrenci: Veli
Mühendislik Fakültesi 3 . ögrenci: Ayse
Tip Fakültesi 1 . ögrenci: Talat
Tip Fakültesi 2 . ögrenci: Zeynep
Tip Fakültesi 3 ögrenci:Ece
"""

students = ["Ali", "Veli", "Ayse", "Talat", "Zeynep", "Ece"]

for index, student in enumerate(students):
    if index < 3:
        print("Mühendislik Fakültesi {}. ögrenci: {}".format(index+1, student))
    else:
        print("Tip Fakültesi {}. ögrenci: {}".format(index-2, student))
        
""" 
Görev 7: Asagida 3 adet liste verilmistir. Listelerde sirasi ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadir. 
Zip kullanarak ders bilgilerini bastiriniz.

ders_ kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3,4,2,4]
kontenian = [30,75,150,25]

Beklenen Çıktı:
Kredisi 3 olan CMP1005 kodlu dersin kontenjanı 30 kisidir.
Kredisi 4 olan PSY1001 kodlu dersin kontenjanı 75 kisidir.
Kredisi 2 olan HUK1005 kodlu dersin kontenjanı 150 kisidir.
Kredisi 4 olan SEN2204 kodlu dersin kontenjanı 25 kisidir.
"""

course_code = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3,4,2,4]
kontenian = [30,75,150,25]

ders_bilgileri = list(zip(course_code, kredi, kontenian))

for i in ders_bilgileri: 
    print(f"Kredisi {i[1]} olan {i[0]} kodlu dersin kontenjanı {i[2]} kisidir.")
    
""" 
Görev 8: Asagida 2 adet set verilmistir. Sizden istenilen eger 1. küme 2. kümeyi kapsiyor ise ortak elemanlarineger kapsamiyor ise 2. 
kümenin 1. kümeden farkini yazdiracak fonksiyonu tanimlamaniz beklenmektedir.

kumel = set( ["data", "python"])
kume2 = set( ['data", "function", "qcut", "lambda", "python", "miuul")

Beklenen Çıktı:
 {'function', 'qcut', 'lambda', 'miuul'}
"""

kume1 = set( ["data", "python"])
kume2 = set( ["data", "function", "qcut", "lambda", "python", "miuul"])

kume1.issuperset(kume2) # False

print(kume2.difference(kume1)) # {'function', 'qcut', 'lambda', 'miuul'}