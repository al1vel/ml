import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    # сортировка по признаку
    order = np.argsort(feature_vector)
    f = feature_vector[order]
    y = target_vector[order]

    # если признак константный, сплитов нет
    if np.all(f == f[0]):
        return np.array([]), np.array([]), None, None

    # возможные места изменения значения признака
    diffs = f[1:] != f[:-1]
    if not np.any(diffs):
        return np.array([]), np.array([]), None, None

    # пороги – средние между соседями
    thresholds = (f[:-1] + f[1:]) / 2
    thresholds = thresholds[diffs]

    n = len(y)

    # префиксные суммы количества классов y==1 и y==0
    # prefix_pos[i] = число единиц среди y[:i]
    prefix_pos = np.cumsum(y == 1)
    prefix_neg = np.cumsum(y == 0)

    # индексы, соответствующие выбранным threshold (где diffs==True)
    idx = np.where(diffs)[0]

    # размеры левого и правого поддеревьев
    left_size_pos = prefix_pos[idx]
    left_size_neg = prefix_neg[idx]
    left_size = left_size_pos + left_size_neg

    right_size_pos = prefix_pos[-1] - left_size_pos
    right_size_neg = prefix_neg[-1] - left_size_neg
    right_size = right_size_pos + right_size_neg

    # исключаем пороги с пустым левым или правым поддеревом
    valid = (left_size > 0) & (right_size > 0)

    if not np.any(valid):
        return np.array([]), np.array([]), None, None

    thresholds = thresholds[valid]
    left_size_pos = left_size_pos[valid]
    left_size_neg = left_size_neg[valid]
    left_size = left_size[valid]
    right_size_pos = right_size_pos[valid]
    right_size_neg = right_size_neg[valid]
    right_size = right_size[valid]

    # доли классов в поддеревьях
    p_left_1 = left_size_pos / left_size
    p_left_0 = left_size_neg / left_size
    p_right_1 = right_size_pos / right_size
    p_right_0 = right_size_neg / right_size

    # энтропия Джини
    H_left = 1 - p_left_1 ** 2 - p_left_0 ** 2
    H_right = 1 - p_right_1 ** 2 - p_right_0 ** 2

    # критерий Q(R)
    ginis = -(left_size / n) * H_left - (right_size / n) * H_right

    # выбираем лучший Gini (максимальный)
    best_idx = np.argmax(ginis)
    gini_best = ginis[best_idx]
    threshold_best = thresholds[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        # критерий останова — все одного класса
        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best = None
        threshold_best = None
        gini_best = None
        split = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            # --- обработка вещественных ---
            if feature_type == "real":
                feature_vector = sub_X[:, feature]

            # --- обработка категориальных ---
            else:
                values = np.unique(sub_X[:, feature])
                # сортировка категорий по P(y=1 | category)
                clicks = Counter(sub_X[sub_y == 1, feature])
                counts = Counter(sub_X[:, feature])

                ratio = {v: clicks[v] / counts[v] if counts[v] > 0 else 0
                         for v in values}

                sorted_categories = sorted(values, key=lambda v: ratio[v])
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])

            # найдём сплит
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None:
                continue

            if (gini_best is None) or (gini > gini_best):
                gini_best = gini
                feature_best = feature

                if feature_type == "real":
                    threshold_best = threshold
                    split = feature_vector < threshold
                else:
                    # категории, которые идут до threshold
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]
                    split = np.isin(sub_X[:, feature], threshold_best)

        # если не нашли сплит → лист
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # создаём внутренний узел
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"] = {}
        node["right_child"] = {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, f"_{key}", value)
        return self
