
def clean_text(str):
    alpha = 'abcdefghijklmnopqrstuvwxyz '
    str = str.lower()
    n_str = ''
    for s in str:
        if s in alpha:
            n_str += s
    return n_str

def count_words(sentance):
    # sentence is a list of words
    c = dict()
    for i in set(sentance):
        c[i] = sentance.count(i)
    return c

with open('text.txt', 'r') as file:
    str = clean_text(file.read())
    words = str.split(' ')

count = count_words(words)
sorted_list = sorted(count.keys(), key=lambda x: count.get(x), reverse=True)

print(f"Most frequent word is '{sorted_list[0]}' which is repeated {count.get(sorted_list[0])} times")