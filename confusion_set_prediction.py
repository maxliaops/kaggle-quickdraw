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
        _, mp = torch.tensor(main_predictions[i]).topk(3, dim=0)

        cs_idx = category_confusion_set_mapping[mp[0]]
        _, csp = torch.tensor(confusion_set_predictions[cs_idx][i]).topk(3, dim=0)

        if set(categories[mp]) == set(confusion_sets[cs_idx][csp]) \
                and catpred(categories[mp]) != catpred(confusion_sets[cs_idx][csp]):
            words.append(catpred(confusion_sets[cs_idx][csp]))
            order_mismatch += 1
        else:
            words.append(catpred(categories[mp]))

        if set(categories[mp]) == set(confusion_sets[cs_idx][csp]):
            set_match += 1

        if set(categories[mp]) != set(confusion_sets[cs_idx][csp]):
            set_mismatch += 1

        if catpred(categories[mp]) != catpred(confusion_sets[cs_idx][csp]):
            mismatch += 1

        if catpred(categories[mp]).split(" ")[0] != catpred(confusion_sets[cs_idx][csp]).split(" ")[0]:
            first_mismatch += 1

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
