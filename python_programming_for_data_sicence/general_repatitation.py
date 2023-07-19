a = "hasan basri akcan"

def string_changer(string):
    new_string = ""
    for i in range(len(string)):
        if i % 2 == 0:
            new_string += string[i].upper()
        else:
            new_string += string[i].lower()
    return new_string
print(string_changer(a))


students = ["ali", "veli", "deli", "ayşe"]
def divide_students(students):
    even = []
    odd = []
    for i in range(len(students)):
        if i % 2 == 0:
            even.append(students[i])
        else:
            odd.append(students[i])
    return [even, odd]
print(divide_students(students))

def altenating_with_enumerate(string):
    new_string = ""
    
    for index, student in enumerate(string):
        if index % 2 == 0:
            new_string += student.upper()
        else:
            new_string += student.lower()
    return new_string
print(altenating_with_enumerate(a))