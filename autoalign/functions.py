import torch
import autoalign.utils

from builtins import sum as builtin_sum
from functools import reduce


def identity(x):
    return x


def pow(x, n):
    return x ** n


def square(x):
    return pow(x, 2)


def pow3(x):
    return pow(x, 3)


def pow4(x):
    return pow(x, 4)


def max(x):
    x = autoalign.utils.to_tensor(x)
    return torch.max(x, 0)[0]


def sum(x):
    return builtin_sum(x)


def mul(x):
    return reduce(lambda x, y: x * y, x)


def mean(x):
    x = autoalign.utils.to_tensor(x)
    return torch.mean(x, 0)


def str_concat(l):
    return " ".join(l)


def list_concat(l):
    return sum(l, [])


def softmax(*args, dim=-1, **kwargs):
    return torch.nn.functional.softmax(*args, dim=dim, **kwargs)
