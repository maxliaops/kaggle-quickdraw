import numpy as np

from utils import read_lines

confusion = np.load("./confusion/confusion.npy")
categories = np.array(read_lines("../../quickdraw/categories.txt"))

for c in range(confusion.shape[0]):
    category_count = confusion[c, :].sum()
    if category_count != 0:
        confusion[c, :] /= category_count

for c in range(confusion.shape[0]):
    confusion[c, c] = 0

count = 0
cats = list()
groups = np.array(range(confusion.shape[0]))
for c1 in range(confusion.shape[0]):
    c2 = np.argmax(confusion[c1, :])
    s = confusion[c1, c2] + confusion[c2, c1]
    if s > 0.0:
        cats.append(c1)
        cats.append(c2)
        g1 = groups[c1]
        g2 = groups[c2]
        groups[groups == g1] = g2

for i, g in enumerate(np.unique(groups)):
    groups[groups == g] = i

print(count)
print(len(cats))
print(len(np.unique(groups)))
xxx = 0
sss = 0
for c in range(confusion.shape[0]):
    if (groups == c).sum() > 1:
        print(categories[groups == c].tolist())
        xxx += 1
        sss += (groups == c).sum()
print(xxx)
print(sss)
m = {i: g for i, g in enumerate(groups)}
print(m)
