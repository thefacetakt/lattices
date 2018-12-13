from typing import List, Set, Dict
from enum import Enum

import numpy as np


class GeneratorFeatures(Enum):
    POS_POWER = 0
    POS_SUPPORT = 1
    POS_CONFIDENCE = 2
    NEG_POWER = 3
    NEG_SUPPORT = 4
    NEG_CONFIDENCE = 5

FEATURE_TYPES = 3

def calculate_power(elements: List[Set], features: Set) -> List[float]:
    return [len(element & features) for element in elements]


def calculate_support(elements: List[Set], features: Set) -> List[float]:
    result = []
    for element in elements:
        intersection = (element & features)
        result.append(np.mean(
            [(intersection < other_element) for other_element in elements]))
    return result


def calculate_confidence(elements: List[Set], neg_elements: List[Set], features: Set) -> List[float]:
    result = []
    for element in elements:
        intersection = (element & features)
        result.append(np.mean(
            [(intersection < other_element) for other_element in neg_elements]))
    return result


def calculate_generator_features(
        positive_elements: List[Set],
        negative_elements: List[Set],
        features: Set) -> Dict[GeneratorFeatures, List[float]]:
    return {
        GeneratorFeatures.POS_POWER: calculate_power(positive_elements, features),
        GeneratorFeatures.NEG_POWER: calculate_power(negative_elements, features),
        GeneratorFeatures.POS_SUPPORT: calculate_support(positive_elements, features),
        GeneratorFeatures.NEG_SUPPORT: calculate_support(negative_elements, features),
        GeneratorFeatures.POS_CONFIDENCE: calculate_confidence(positive_elements, negative_elements, features),
        GeneratorFeatures.NEG_CONFIDENCE: calculate_confidence(negative_elements, positive_elements, features),
    }

def calculate_aggreagated_features(generator_features: Dict[GeneratorFeatures, List[float]]) -> List[float]:
    result = []
    for function in [np.min, np.max, np.median, np.mean]:
        for g_value in range(2 * FEATURE_TYPES):
            key = GeneratorFeatures(g_value)
            value = generator_features[key]
            if g_value > FEATURE_TYPES:
                result.append(1.0 - function(value))
            else:
                result.append(function(value))
    
    for function in [np.min, np.max, np.median, np.mean]:
        for pos_index in range(FEATURE_TYPES):
            neg_index = pos_index + FEATURE_TYPES
            pos_features = np.array(generator_features[GeneratorFeatures(pos_index)])
            neg_features = np.array(generator_features[GeneratorFeatures(neg_index)])
            result.append(function(pos_features) - function(neg_features))
    return result


def get_feature_names() -> List[str]:
    result = []
    for function in [np.min, np.max, np.median, np.mean]:
        for g_value in range(2 * FEATURE_TYPES):
            key = GeneratorFeatures(g_value)
            key = str(key).split(".")[-1]
            if g_value > FEATURE_TYPES:
                result.append("-{}({})".format(function.__name__, key))
            else:
                result.append("{}({})".format(function.__name__, key))
    for function in [np.min, np.max, np.median, np.mean]:
        for pos_index in range(FEATURE_TYPES):
            neg_index = pos_index + FEATURE_TYPES
            pos_key = str(GeneratorFeatures(pos_index)).split(".")[-1]
            neg_key = str(GeneratorFeatures(neg_index)).split(".")[-1]
            result.append("{}({}-{})".format(function.__name__, pos_key, neg_key))
    return result

def get_features(X_train: List[Set], y_train: List[bool], X_test: List[Set]):
    positive = [x for (x, y) in zip(X_train, y_train) if y]
    negative = [x for (x, y) in zip(X_train, y_train) if not y]
    features = [
        calculate_aggreagated_features(calculate_generator_features(positive, negative, x))
        for x in X_test]
    return features