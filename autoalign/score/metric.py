import torch

import scipy


def cosine_similarity(a, b):
    if len(a.shape) == 1:
        return vector_cosine_similarity(a, b)
    else:
        return matrix_cosine_similarity(a, b)


def vector_cosine_similarity(a, b):
    return scipy.spatial.distance.cosine(a, b)


def matrix_cosine_similarity(a, b):
    """Cosine similarity for matrices
       source: https://stackoverflow.com/questions/50411191/

    Args:
        a: [n, d]
        b: [m, d]

    Return:
        cosine: [n, m]
    """

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res


def heuristic_max_similarity(s1, s2):
    """(same doc as `cosine_similarity`
    """
    # try:
    #     stacked_s2 = torch.stack(s2, 0)
    # except Exception as e:
    #     print(s1, s2)
    #     raise e

    sum_cos = torch.zeros([1])
    for v1 in s1:
        cos_v1_s2 = torch.nn.functional.cosine_similarity(
            v1.expand_as(s2), s2)
        sum_cos += cos_v1_s2.max()

    sum_cos.div_(len(s1))
    return sum_cos
