## Finding Union and Intersection
# SHORTER VERSION

lst1 = set([int(x) for x in input("Enter the list element seperated by ',': ").split(',')])
lst2 = set([int(x) for x in input("Enter the list element seperated by ',': ").split(',')])

print('Union is: ', list(lst1|lst2))
print('Intersection is: ', list(lst1&lst2))