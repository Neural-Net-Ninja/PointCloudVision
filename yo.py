dict_a = {'apple': 1, 'banana': 2, 'cherry': 3}
dict_b = {'apple_pie': 4, 'banana_split': 5, 'mango_sorbet': 7}

not_matched = []
for key_b in dict_b.keys():
    matched = False
    for key_a in dict_a.keys():
        if key_b.startswith(key_a):
            matched = True
            break
    if not matched:
        not_matched.append(key_a)

print(not_matched)  # ['mango_sorbet']

