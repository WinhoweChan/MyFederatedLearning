import time
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class MyFunction:
    def __init__(self):
        pass

    def score(self, global_model, local_models, clients_types, selected_clients, m=0):

        n = len(selected_clients)
        P = generate_orthogonal_matrix(n=m, reuse=True)
        W = generate_orthogonal_matrix(n=n * n, reuse=True)
        Ws = [W[:, e * n: e * n + n][0, :].reshape(-1, 1) for e in range(n)]

        param_diff = []
        param_diff_mask = []

        m_len = len(local_models)

        dw = []
        db = []
        for i, local_model in enumerate(local_models):
            # 计算权重差异
            dw_item = global_model.state_dict()['fc2.weight'].cpu().numpy() - local_model.state_dict()[
                'fc2.weight'].cpu().numpy()
            dw.append(np.linalg.norm(dw_item, axis=-1))
            db_item = global_model.state_dict()['fc2.bias'].cpu().numpy() - local_model.state_dict()[
                'fc2.bias'].cpu().numpy()
            db.append(np.abs(db_item))

        memory = np.sum(dw, axis=0)
        memory += np.sum(db, axis=0)

        # 找到最常出现的两个类别
        max_two_freq_classes = np.argsort(memory)[-2:]

        source_class_1, source_class_2 = max_two_freq_classes
        logger.debug('Potential source: ' + str(source_class_1) + ', ' + str(source_class_2))

        detect_class = source_class_2
        logger.debug('Detect class: ' + str(detect_class))
        start_model_layer_param = list(global_model.state_dict()['fc2.weight'][detect_class].cpu())
        # 计算每个本地模型的权重与全局模型最后一层权重之间的梯度差
        for i in range(m_len):
            end_model_layer_param = list(local_models[i].state_dict()['fc2.weight'][detect_class].cpu())

            gradient = calculate_parameter_gradients(start_model_layer_param, end_model_layer_param)
            gradient = gradient.flatten()

            param_diff.append(gradient)

            X_mask = Ws[i] @ gradient.reshape(1, -1) @ P
            param_diff_mask.append(X_mask)

        Z_mask = sum(param_diff_mask)
        U_mask, sigma, VT_mask = svd(Z_mask)

        G = Ws[0]
        for idx, val in enumerate(selected_clients):
            if idx == 0:
                continue
            G = np.concatenate((G, Ws[idx]), axis=1)

        U = np.linalg.inv(G) @ U_mask
        U = U[:, :2]
        res = U * sigma[:2]
        scores = detect_outliers_kmeans(res)

        logger.debug("Defense result:")
        for i, pt in enumerate(clients_types):
            logger.info(str(pt) + ' scored ' + str(scores[i]))

        # data = StandardScaler().fit_transform(res)
        # draw(data, clients_types, scores)

        # 返回得分列表
        return scores


def generate_orthogonal_matrix(n, reuse=False, block_size=None):
    orthogonal_matrix_cache_dir = 'orthogonal_matrices'
    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    file_name = str(n)
    if block_size is not None:
        file_name += '_blc%s' % block_size

    if reuse and file_name in existing:
        with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        if block_size is not None:
            qs = [block_size] * int(n / block_size)
            if n % block_size != 0:
                qs[-1] += (n - np.sum(qs))
            q = np.zeros([n, n])
            for i in range(len(qs)):
                sub_n = qs[i]
                tmp = generate_orthogonal_matrix(sub_n, reuse=False, block_size=sub_n)
                index = int(np.sum(qs[:i]))
                q[index:index + sub_n, index:index + sub_n] += tmp
        else:
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        if reuse:
            with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(q, f, protocol=4)
        return q


def calculate_parameter_gradients(params_1, params_2):
    return np.array([x for x in np.subtract(params_1, params_2)])


def detect_outliers_kmeans(data, n_clusters=2):
    # 初始化K-means模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # 训练模型
    kmeans.fit(data)
    # 预测聚类标签
    labels = kmeans.predict(data)
    # 计算轮廓系数
    coefficient = silhouette_score(data, kmeans.labels_)
    logger.debug("Silhouette Coefficient：{}", coefficient)
    if coefficient < 0.65:
        return np.ones(len(data), dtype=int)

    scores = labels
    if sum(labels) < len(data) / 2:
        scores = 1 - labels
    else:
        scores = labels

    return scores


def secure_aggregation(xs):
    n = len(xs)
    size = xs[0].shape
    # Step 1 Generate random samples between each other
    perturbations = []
    for i in range(n):
        tmp = []
        for j in range(n):
            tmp.append(my_random(size))
        perturbations.append(tmp)
    perturbations = np.array(perturbations)
    perturbations -= np.transpose(perturbations, [1, 0, 2, 3])
    ys = [xs[i] - np.sum(perturbations[i], axis=0) for i in range(n)]

    results = np.sum(ys, axis=0)
    return results


def my_random(size):
    return np.random.randint(low=-10 ** 5, high=10 ** 5, size=size) + np.random.random(size)


def svd(x):
    m, n = x.shape
    if m >= n:
        return np.linalg.svd(x)
    else:
        u, s, v = np.linalg.svd(x.T)
        return v.T, s, u.T


def draw(data, clients_types, scores):
    SAVE_NAME = str(time.time()) + '.jpg'

    fig = plt.figure(figsize=(34, 12))
    fig1 = plt.subplot(121)
    for i, pt in enumerate(clients_types):
        if pt == 'Good update':
            plt.scatter(data[i, 0], data[i, 1], facecolors='none', edgecolors='black', marker='o', s=800,
                        label="Good update")
        else:
            plt.scatter(data[i, 0], data[i, 1], facecolors='black', edgecolors='black', marker='o', s=800,
                        label="Bad update")

    fig2 = plt.subplot(122)
    for i, pt in enumerate(clients_types):
        if scores[i] == 1:
            plt.scatter(data[i, 0], data[i, 1], color="orange", s=800, label="Good update")
        else:
            plt.scatter(data[i, 0], data[i, 1], color="blue", marker="x", linewidth=3, s=800, label="Bad update")

    plt.grid(False)
    # plt.show()
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)

