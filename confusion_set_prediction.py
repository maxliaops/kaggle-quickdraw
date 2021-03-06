import numpy as np
import pandas as pd
import torch

from utils import read_lines


def catpred(categories):
    return " ".join([c.replace(" ", "_") for c in categories])


def predict():
    input_dir = "./confusion"
    num_confusion_sets = 6

    categories = np.array(read_lines("../../quickdraw/categories.txt"))

    confusion_sets = [np.array(read_lines("{}/confusion_set_{}.txt".format(input_dir, c))) for c in
                      range(num_confusion_sets)]
    category_confusion_set_mapping = np.load("{}/category_confusion_set_mapping.npy".format(input_dir))

    main_predictions = np.load("{}/predictions_main.npy".format(input_dir))
    confusion_set_predictions = [np.load("{}/predictions_cs{}.npy".format(input_dir, c)) for c in
                                 range(num_confusion_sets)]

    words = []
    mismatch = 0
    first_mismatch = 0
    set_mismatch = 0
    set_match = 0
    order_mismatch = 0
    for i in range(len(main_predictions)):
        vmp, mp = torch.tensor(main_predictions[i]).topk(3, dim=0)
        mcats = categories[mp]
        mmap = {c: v for c, v in zip(mcats, vmp)}

        cs_idx = category_confusion_set_mapping[mp[0]]
        vcsp, csp = torch.tensor(confusion_set_predictions[cs_idx][i]).topk(3, dim=0)
        cscats = confusion_sets[cs_idx][csp]
        csmap = {c: v for c, v in zip(cscats, vcsp)}

        if set(mcats) == set(cscats) and catpred(mcats) != catpred(cscats) and cs_idx in [0, 2, 4]:
            if True:
                words.append(catpred(cscats))
            else:
                fmap = {c: mmap[c] + csmap[c] for c in mcats}
                ensemble_words = sorted(fmap, key=lambda k: -fmap[k])
                words.append(catpred(ensemble_words))

            order_mismatch += 1
        else:
            words.append(catpred(mcats))

        if set(mcats) == set(cscats):
            set_match += 1

        if set(mcats) != set(cscats):
            set_mismatch += 1

        if catpred(mcats) != catpred(cscats):
            mismatch += 1

        if catpred(mcats).split(" ")[0] != catpred(cscats).split(" ")[0]:
            first_mismatch += 1

    print(len(main_predictions))
    print(mismatch)
    print(first_mismatch)
    print(set_mismatch)
    print(set_match)
    print(order_mismatch)

    return words


def main():
    df = pd.read_csv("../../quickdraw/test_simplified.csv", index_col="key_id")
    df["word"] = predict()
    df.to_csv("./submission_confusion_sets.csv", columns=["word"])


if __name__ == "__main__":
    main()
