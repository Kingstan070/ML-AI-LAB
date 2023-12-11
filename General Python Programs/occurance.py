# Count the occurrence of each word in a given sentence

def count_words(sentance):
    # sentence is a list of words
    c = dict()
    for i in set(sentance):
        c[i] = sentance.count(i)
    return c

sentance = input('Enter the sentance > ').lower().split(' ')

print('Count of each words in the sentance: ')
for i,j in count_words(sentance).items():
    print(f'{i} > {j}')

# The program is not optimized, 
# if there is any ',' or '.' or any other punctuations 
# will result in wrong output

# Solution is to replace the those puntuations with ''