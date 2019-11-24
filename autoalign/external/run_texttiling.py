#!/usr/bin/env python
import os
import json

from autoalign.legacy.segment_job import JobSegmenter
from pathos.multiprocessing import ProcessingPool as Pool
from copy import deepcopy

from grid_search import grid_search_params
from texttiling import read_ctm, texttiling_text

from run_c99 import is_separator, seg_count, SEPARATOR_STR, root, src_ext, ref_ext
texttiling_ext = src_ext + ".ttseg"
result_ext = texttiling_ext + ".results"

empty_params = {
    "k": [],
    "w": [],
    "smoothing_width": [],
    "smoothing_rounds": []
}

exps = []

exp0 = deepcopy(empty_params)
exp0["k"] = [5, 10, 20, 40]
exp0["w"] = [5, 20, 40, 60, 80]
exp0["smoothing_width"] = [5, 10, 15]
exp0["smoothing_rounds"] = [2, 5, 10]
exps.append(exp0)

exp1 = deepcopy(empty_params)
exp1["k"] = [5, 10, 15, 20, 30, 40]
exp1["w"] = [5, 20, 40, 50, 60, 80]
exp1["smoothing_width"] = [2, 5, 7, 10, 12, 15, 20]
exp1["smoothing_rounds"] = [1, 2, 5, 7, 10, 15]
exps.append(exp1)

# exp0 with other width/rounds
exp2 = deepcopy(empty_params)
exp2["k"] = [5, 10, 15, 20, 30, 40]
exp2["w"] = [5, 20, 40, 50, 60, 80]
exp2["smoothing_width"] = [2, 7, 12, 20]
exp2["smoothing_rounds"] = [1, 3, 7, 15]
exps.append(exp2)


def process_src(source_path, n_exp=0):
    ref_path = source_path.replace(src_ext, ref_ext)
    seg_n = seg_count(ref_path)
    out_path = source_path.replace(src_ext, texttiling_ext)

    with open(source_path) as f:
        # double line separation of segments
        source_txt = "\n\n".join([_.strip() for _ in f])

    params = grid_search_params(exps[n_exp])

    ftext = "Failed"
    for i, p in enumerate(params):
        print("\tProcessing param [%d/%d] (%s)" %
              (i, len(params), source_path))
        o = texttiling_text(source_txt, **p)
        if len(o) == seg_n:
            o = [_.strip().replace("\n\n", "\n") for _ in o]
            o = [_ for _ in o if len(_) > 0]
            o = ("\n%s\n" % SEPARATOR_STR).join(o)
            param_str = ", ".join(["%s=%s" % (k, str(p[k])) for k in [
                                  "k", "w", "smoothing_width", "smoothing_rounds"]])
            ftext = "%s %s\n" % (SEPARATOR_STR, param_str) + o
            print("\t!!! MATCH")
            break
    with open(out_path, 'w') as f:
        print(ftext, file=f)


def process_src_args(args):
    return process_src(*args)


def run_texttiling(root_path, n_exp=0, n_thread=4):

    sources = [
        os.path.join(root_path, f)
        for f in os.listdir(root_path)
        if f.endswith(src_ext)
    ]

    print("Found '%d' sources" % len(sources))
    filtered_sources = [
        _
        for _ in sources
        if not os.path.exists(_.replace(src_ext, texttiling_ext))
    ]
    print("Processing '%d' files" % len(filtered_sources))
    with Pool(processes=n_thread) as pool:
        pool.map(process_src_args, [(source, n_exp,)
                                    for source in filtered_sources])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-root", default=root, type=str)
    parser.add_argument("-n_thread", type=int, default=4)
    parser.add_argument("-n", type=int, default=0)
    args = parser.parse_args()

    run_texttiling(args.root, args.n, args.n_thread)
