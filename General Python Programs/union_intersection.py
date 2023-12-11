## Finding Union and Intersection

def union_intersection(lst1, lst2):
    union = list(set(lst1) | set(lst2))
    intersection = list(set(lst1) & set(lst2))
    return union, intersection

lst1 = []
lst2 = []

n = int(input("Enter the lenght of first list: "))
for i in range(n):
    element = int(input(f'lst1[{i}]> '))
    lst1.append(element)

n = int(input("Enter the lenght of second list: "))
for i in range(n):
    element = int(input(f'lst2[{i}]> '))
    lst2.append(element)

u, i = union_intersection(lst1, lst2)
print('Union is:', u)
print('Intersection is:', i)