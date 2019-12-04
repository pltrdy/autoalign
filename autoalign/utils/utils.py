import torch
import numpy as np
import autoalign


def describe(element):
    if isinstance(element, np.ndarray):
        print("np.ndarray[%s]" % str(element.shape))

    elif isinstance(element, torch.Tensor):
        return "tensor[%s]" % str(list(element.size()))

    elif isinstance(element, list):
        return "list#%d[%s]" % (len(element), "empty"
                                if len(element) == 0 else describe(element[0]))

    return "%s: '%s'" % (str(type(element)), str(element))


def assert_size(tensor, shape):
    tsize = list(tensor.size())

    assert all([s == -1 or s == t for t, s in zip(tsize, shape)]), \
        "Tensor size mismatch %s, expected %s" % (str(tsize), str(shape))


def to_tensor(x):
    if isinstance(x, list):
        x = torch.stack(x, 0)
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x)
    return x


def is_fct(o):
    return hasattr(o, '__call__')


def to_fct(fct):
    """Return a function of autoalign.functions from it's name (if fct is str)
       or fct if it's a function itself

    Args:
        fct(object): function, or the function name

    Returns:
        a callable function

    Raise:
        ValueError: fct isn't a function nor a str
    """
    if is_fct(fct):
        return fct
    elif isinstance(fct, str):
        return getattr(autoalign.functions, fct)
    raise ValueError("No function associated with '%s' type" % str(type(fct)))


def to_fct_or_none(fct):
    if fct is None:
        return None
    return to_fct(fct)
