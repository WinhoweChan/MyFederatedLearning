import torch


def Krum(updates, f, multi=False):
    # 使用 Krum 算法来选择可信度最高的更新数据

    print("---------running krum--------------")

    # 获取更新数据的数量
    n = len(updates)

    # 将更新数据转换为一维张量
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]

    # 创建一个大小为 [n, 更新数据长度] 的张量
    updates_ = torch.empty([n, len(updates[0])])

    # 将每个更新数据的一维张量复制到 updates_ 张量的对应行中
    for i in range(n):
        updates_[i] = updates[i]

    # 计算选择前 k 个更新数据所需的阈值 k
    k = n - f - 2

    # 计算每个更新数据与其他更新数据之间的欧几里得距离
    cdist = torch.cdist(updates_, updates_, p=2)

    # 选择距离其他更新数据最远的前 k 个更新数据
    dist, idxs = torch.topk(cdist, k, largest=False)

    # 计算选择的更新数据之间的距离之和
    dist = dist.sum(1)

    # 将距离之和按升序排序，并返回索引列表
    idxs = dist.argsort()

    # 如果 multi 标志为 True，则返回前 k 个可信度最高的更新数据的索引
    if multi:
        return idxs[:k]
    # 否则，返回可信度最高的更新数据的索引
    else:
        return idxs[0]
