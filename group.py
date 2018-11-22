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

groups = np.array(range(confusion.shape[0]))
for c1 in range(confusion.shape[0]):
    c2 = np.argmax(confusion[c1, :])
    s = confusion[c1, c2] + confusion[c2, c1]
    if s > 0.0:
        g1 = groups[c1]
        g2 = groups[c2]
        groups[groups == g1] = g2

for i, g in enumerate(sorted(np.unique(groups))):
    groups[groups == g] = i

for g in range(np.max(groups) + 1):
    print(categories[groups == g].tolist())

m = {i: g for i, g in enumerate(groups)}
print(m)
