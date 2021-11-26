'''
Classify a sample into the Iris dataset
'''
import numpy as np
from sklearn import datasets
from scipy import stats


def distance(sample1, sample2):
    diff = [(sample2[i]-sample1[i])**2 for i in range(len(sample1))]
    return sum(diff)


def knn_classify(samples, k, new_point):
    dis = [distance(s, new_point) for s in samples]
    knn_targets = iris_target[np.argsort(dis)][0:k]
    return stats.mode(knn_targets)


if __name__ == "__main__":
    iris_data = datasets.load_iris()
    iris_samples = iris_data.data
    iris_target = iris_data.target
    new_sample = [.1, 2, 3, 8]
    result = knn_classify(iris_samples, 5, new_sample)[0]
    print("The sample belongs to class: %d" % result[0] )