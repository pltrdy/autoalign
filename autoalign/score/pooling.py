import torch


def sentence_sum_pooling(embeddings):
    """
    Args:
        embeddings(list[tensor]) list of word embeddings [dim]

    Return:
        sentence_embedding [dim]
    """
    return sum(embeddings)


def sentence_max_pooling(embeddings):
    t = torch.stack(embeddings, -1)
    return t.max(-1)[0]


def sentence_mean_pooling(embeddings):
    t = torch.stack(embeddings, -1).float()
    return t.mean(-1)


def no_pooling(embeddings):
    return embeddings
