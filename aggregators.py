def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from collections import defaultdict
from features import get_features
from typing import Dict, List, Set, Tuple
from pathos.multiprocessing import ProcessingPool
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np

FOLDS = 10
NUM_FEATURES = 9 * 4
THRESHOLDS = 10

def get_metrics(y_test, y_score):
    return {
        "accuracy": accuracy_score(y_test, y_score),
        "f1_score": f1_score(y_test, y_score),
        "precision": precision_score(y_test, y_score),
        "recall": recall_score(y_test, y_score)
    }


def get_best_ds_aggregator(X: List[Set], y: List[int]) -> Tuple[float, Tuple[int, float]]:
    class Model(object):
        def __init__(self, index: int, threshold: float):
            self.index = index
            self.threshold = threshold
        
        def fit(self, X_train: List[Set], y_train: List[int]):
            self.X_train = X_train
            self.y_train = y_train
        
        def predict(self, X_test: List[Set]) -> List[bool]:
            features = get_features(self.X_train, self.y_train, X_test)
            return (np.array(features))

    kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)
    params2scores = defaultdict(lambda: defaultdict(list))

    def k_fold_iteration(args):
        params2score = defaultdict(dict)
        train_index, test_index = args
        X_train = [X[ind] for ind in train_index]
        y_train = [y[ind] for ind in train_index]
        X_test = [X[ind] for ind in test_index]
        y_test = [y[ind] for ind in test_index]
        features = get_features(X_train, y_train, X_test)
        features = np.array(features)
        
        for index in range(NUM_FEATURES):
            threshold_range = np.linspace(np.min(features[:, index]), np.max(features[:, index]), THRESHOLDS)
            for threshold in threshold_range:
                y_score = features[:, index] > threshold
                score = f1_score(y_test, y_score)
                params2score[(index, threshold)] = get_metrics(y_test, y_score)
        return params2score
    
    with tqdm(total=FOLDS) as pbar:
        with ProcessingPool(FOLDS) as pool:
            for params2score in pool.imap(k_fold_iteration, kf.split(X)):
                for params, score in params2score.items():
                    for score_name, score_value in score.items():
                        params2scores[params][score_name].append(score_value)
                pbar.update(1)

    best_params = sorted(list(params2scores), key=lambda params: -np.mean(params2scores[params]["accuracy"]))
    best_score = {key: np.mean(value) for key, value in params2scores[best_params[0]].items()}
    return best_score, best_params

def score(index, threshold, X_train, y_train, X_test):
    features = get_features(X_train, y_train, X_test)
    return np.array(features)[:, index] > threshold