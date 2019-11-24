#!/usr/bin/env python
import os
import pickle


def aligned_stats(dir_path):
    aligned_path = [os.path.join(dir_path, _)
                    for _ in os.listdir(dir_path)
                    if _.endswith(".aligned.pckl")]

    tot_n_ctm = 0
    tot_w_ctm = 0

    tot_n_doc = 0
    tot_w_doc = 0
    for path in aligned_path:
        with open(path, 'rb') as f:
            d = pickle.load(f)

        print(d["final"]["ctm"][0])
        tot_n_ctm += len(d["final"]["ctm"])
        tot_n_doc += len(d["final"]["doc"])
        tot_w_ctm += sum([len(_["text"].split()) for _ in d["final"]["ctm"]])
        tot_w_doc += sum([len(_["text"].split()) for _ in d["final"]["doc"]])
    print("tot n ctm: %d" % tot_n_ctm)
    print("tot w ctm: %d" % tot_w_ctm)
    print("avg: %.3f" % (tot_w_ctm / tot_n_ctm))
    print("avg: %.3f" % (tot_w_ctm / tot_n_doc))
    print("---")
    print("tot n doc: %d" % tot_n_doc)
    print("tot w doc: %d" % tot_w_doc)
    print("avg: %.3f" % (tot_w_doc / tot_n_doc))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "-d", required=True)

    args = parser.parse_args()
    aligned_stats(args.dir)
