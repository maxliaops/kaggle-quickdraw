import numpy as np
import pandas as pd

from utils import read_lines


def to_word(categories):
    return " ".join([c.replace(" ", "_") for c in categories])


def predict():
    input_dir = "/Users/omallo/Downloads"

    categories = np.array(read_lines("../../quickdraw/categories.txt"))

    predictions1 = np.load("{}/predictions_seresnext50.npy".format(input_dir))
    predictions2 = np.load("{}/predictions_drn_d_105.npy".format(input_dir))

    words = []
    for p1, p2 in zip(predictions1, predictions2):
        p = (p1 + p2) / 2
        c = np.argsort(-p)[:3]
        words.append(to_word(categories[c]))

    return words


def main():
    df = pd.read_csv("../../quickdraw/test_simplified.csv", index_col="key_id")
    df["word"] = predict()
    df.to_csv("./submission_ensemble.csv", columns=["word"])


if __name__ == "__main__":
    main()
