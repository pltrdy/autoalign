#!/usr/bin/env python
import argparse
import torch
import os


def load_vectors(word2vec_path, n_vectors, encoding='utf-8', output_path=None,
                 verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    tensors = []
    itos = []
    with open(word2vec_path, 'rb') as w2v_file:
        line = w2v_file.readline()
        n_vec, n_dim = [int(_) for _ in line.split()[:2]]

        vprint("Reading vector file: %d vectors of %d dimensions"
               % (n_vec, n_dim))

        if n_vectors > 0:
            n_vectors = min(n_vectors, n_vec)
        else:
            n_vectors = n_vec
        vprint("n_vectors: %d" % n_vectors)

        step = n_vectors / 100
        bigstep = n_vectors / 10
        for i, line in enumerate(w2v_file):
            ii = i + 1
            if i == n_vectors:
                break

            if i % step == step - 1:
                if i % bigstep == bigstep - 1:
                    endline = "\n"
                else:
                    endline = ""

                vprint("\rLoading vector: %d/%d\t%d%%"
                       % (ii, n_vectors, int(ii // step)), end=endline)

            elts = line.split()
            word, values = elts[0], [float(_) for _ in elts[1:]]
            assert len(values) == n_dim

            itos += [word.decode(encoding)]
            tensor = torch.Tensor(values)
            tensors += [tensor]

    final_tensor = torch.stack(tensors, 0)
    vprint("\n****")
    vprint("Final tensor: %s" % str(final_tensor.size()))

    obj = {"embeddings": final_tensor, "itos": itos}
    if output_path is not None:
        vprint("Saving embeddings / vocab to '%s'" % output_path)
        torch.save(obj, output_path)
    vprint("\nDone.")

    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Load word2vec (text format)
                       to tensor representation""")
    parser.add_argument("--file", "-f", required=True,
                        help="word2vec (text) file")
    parser.add_argument("--output", "-o", default="word2vec.pt",
                        help="Output file")
    parser.add_argument('--n_vectors', '-n', default=150000, type=int,
                        help="Number of vectors to consider (-1 means all)")
    args = parser.parse_args()
    main(args.file, args.n_vectors, args.output, verbose=True)
