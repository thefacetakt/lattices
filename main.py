from typing import Tuple, List, Set
from aggregators import get_best_ds_aggregator, get_metrics, score
from features import get_feature_names
from sklearn.model_selection import train_test_split
import sys

def read_ttt_dataset(filepath) -> Tuple[List[Set], List[Set], List[bool], List[bool]]:
    X = []
    y = []
    with open(filepath) as file:
        for line in file:
            data = line.strip().split(',')
            y.append(data[-1] == 'positive')
            data = data[:-1]
            x = set()
            for i, element in enumerate(data):
                x.add("{}_{}".format(i, element))
            X.append(x)

    X, X_val, y, y_val = train_test_split(X, y, random_state=41, test_size=0.1)
    return X, X_val, y, y_val

def main():
    feature_names = get_feature_names()
    X, X_val, y, y_val = read_ttt_dataset("data/tic-tac-toe.data")
    best_score, params = get_best_ds_aggregator(X, y)
    index, threshold = params[0]

    print("Top rule:", feature_names[index], "> {:.4}".format(threshold))

    for score_name, score_value in best_score.items():
        print("{}: {:.4f}".format(score_name, score_value))
    y_score = score(index, threshold, X, y, X_val)
    for score_name, score_value in get_metrics(y_val, y_score).items():
        print("val_{}: {:.4f}".format(score_name, score_value))

    print("Top 10 rules:")
    for i in range(10):
        print(feature_names[params[i][0]], "> {:.4}".format(params[i][1]))

if __name__ == '__main__':
    main()