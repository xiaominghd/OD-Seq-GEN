import numpy as np
import scipy.stats


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def distance(x, y):
    return abs(x // 50 - y // 50) + abs(x % 50 - y % 50)


def distribution(arr, min_val, max_val, bin=10):
    arr = np.array(arr)
    arr = (arr - min_val) / (max_val - min_val)
    dist, base = np.histogram(arr, np.arange(0, 1, 1. / bin))

    return np.array(dist) / sum(dist)


class JSD_Metrix(object):

    def __init__(self, inp, target):

        self.inp = inp
        self.target = target

    def get_JSD_distance(self):
        p = []
        q = []
        for i in range(len(self.inp)):
            if len(self.inp[i]) == 1:
                p.append(0)
                continue
            for j in range(len(self.inp[i]) - 1):
                p.append(distance(self.inp[i][j][0], self.inp[i][j + 1][0]))

        for i in range(len(self.target)):
            if len(self.target[i]) == 1:
                q.append(0)
                continue
            for j in range(len(self.target[i]) - 1):
                q.append(distance(self.target[i][j][0], self.target[i][j + 1][0]))
        min_val = min(min(p), min(q))
        max_val = max(max(p), max(q))
        p = distribution(p, min_val, max_val, bin=20)
        q = distribution(q, min_val, max_val, bin=20)

        return JS_divergence(p, q)

    def get_JSD_trajlen(self):

        p = []
        q = []

        for i in range(len(self.inp)):
            p.append(len(self.inp[i]))

        for i in range(len(self.target)):
            q.append(len(self.target[i]))

        min_val = min(min(p), min(q))
        max_val = max(max(p), max(q))
        p = distribution(p, min_val, max_val, bin=20)
        q = distribution(q, min_val, max_val, bin=20)

        return JS_divergence(p, q)

    def get_JSD_Loc(self):
        p = []
        q = []

        for i in range(len(self.inp)):
            for j in range(len(self.inp[i])):
                p.append(self.inp[i][j][0])

        for i in range(len(self.target)):
            for j in range(len(self.target[i])):
                q.append(self.target[i][j][0])

        min_val = min(min(p), min(q))
        max_val = max(max(p), max(q))
        p = distribution(p, min_val, max_val, bin=500)
        q = distribution(q, min_val, max_val, bin=500)
        return JS_divergence(p, q)


    def get_JSD_duration(self):

        p = np.zeros([24])
        q = np.zeros([24])

        for i in range(len(self.inp)):
            for j in range(len(self.inp[i])):
                p[self.inp[i][j][2]] += 1

        for i in range(len(self.target)):
            for j in range(len(self.target[i])):
                q[self.target[i][j][2]] += 1

        p = np.array(p / sum(p))
        q = np.array(q / sum(q))

        return JS_divergence(p, q)

    def get_JSD_start(self):
        p = np.zeros([24])
        q = np.zeros([24])
        for i in range(len(self.inp)):
            for j in range(len(self.inp[i])):
                p[self.inp[i][j][1]] += 1
        for i in range(len(self.target)):
            for j in range(len(self.target[i])):
                q[self.target[i][j][1]] += 1
        p = np.array(p / sum(p))
        q = np.array(q / sum(q))

        return JS_divergence(p, q)

    def get_JSD_end(self):
        p = np.zeros([24])
        q = np.zeros([24])
        for i in range(len(self.inp)):
            for j in range(len(self.inp[i])):
                p[int((self.inp[i][j][1] + self.inp[i][j][2]) / 2)] += 1
        for i in range(len(self.target)):
            for j in range(len(self.target[i])):
                q[int((self.target[i][j][1] + self.target[i][j][2]) / 2)] += 1
        p = np.array(p / sum(p))
        q = np.array(q / sum(q))

        return JS_divergence(p, q)

    def get_redius(self):
        p = []
        q = []
        for i in range(len(self.inp)):
            if len(self.inp[i]) == 1:
                p.append(0)
                continue
            for j in range(len(self.inp[i]) - 1):
                p.append(max(0, distance(self.inp[i][0][0], self.inp[i][j][0])))

        for i in range(len(self.target)):
            if len(self.target[i]) == 1:
                q.append(0)
                continue
            for j in range(len(self.target[i]) - 1):
                q.append(max(0, distance(self.target[i][0][0], self.target[i][j][0])))
        min_val = min(min(p), min(q))
        max_val = max(max(p), max(q))
        p = distribution(p, min_val, max_val, bin=20)
        q = distribution(q, min_val, max_val, bin=20)

        return JS_divergence(p, q)
