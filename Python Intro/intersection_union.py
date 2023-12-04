def union_intersection(lst1, lst2): 
    union = list(set(lst1) | set(lst2)) 
    intersection = list(set(lst1) & set(lst2))
    return union, intersection


nums1 = [1,2,3,4,5]
nums2 = [3,4,5,6,7,8]
print("Original lists:")
print(nums1)
print(nums2)
result = union_intersection(nums1, nums2)
print("\nUnion of said two lists:")
print(result[0])
print("\nIntersection of said two lists:")
print(result[1])
colors1 = ["Red", "Green", "Blue"]
colors2 = ["Red", "White", "Pink", "Black"] 
print("Original lists:")
print(colors1)
print(colors2)
result = union_intersection(colors1, colors2)
print("\nUnion of said two lists:")
print(result[0])
print("\nIntersection of said two lists:")
print(result[1])
