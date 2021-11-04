import pandas as pd
import random


file = 'data/imdb_sup_test.txt'
with open(file, 'r', encoding='utf-8') as f:
    data = f.readlines()
    columns = data.pop(0)
    random.shuffle(data)
    data = [columns] + data


file = 'data/sani.txt'
with open(file, 'w', encoding='utf-8') as f:
    for i, l in enumerate(data):
        # column 1 + data 500
        if i == 501:
            break
        f.write(l)