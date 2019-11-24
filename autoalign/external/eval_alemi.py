#!/usr/bin/env python
import os
import json
import subprocess

from nltk.metrics import windowdiff

SEPARATOR_STR = "=========="

src_ext = ".aligned.pckl.txt"
ref_ext = src_ext + ".seg.ref"
alemi_ext = src_ext + ".alemiseg"
result_ext = alemi_ext + ".results"


def is_separator(line):
    return line.startswith(SEPARATOR_STR)


def seg_count(path):
    # starts at -1 since there's both begin/end separator
    count = -1
    with open(path) as f:
        for line in f:
            if is_separator(line):
                count += 1
    return count


def compare(hyp, ref):
    def _read_lines(path):
        with open(path) as f:
            lines = [_.replace("START ", "").replace(
                "END", "").strip() for _ in f]
            lines = [_ for _ in lines if len(_) > 0]
        return lines

    hyp_lines = _read_lines(hyp)
    ref_lines = _read_lines(ref)

    tot_words = 0
    aligned_words = 0

    tot_seg = 0
    aligned_seg = 0

    # an alignment notation as list of tuple (align, expected)
    alignment = []

    def _numericalize_segments(lines, start_at=1):
        """From lines w/ separator, returns
           [(line, seg_id),]
        """
        numericalized = []
        borders = []
        count = 0
        # ignore first, a separator in any case
        for line in lines[start_at:]:
            if is_separator(line):
                count += 1
                borders += [1]
            else:
                n_line = (line, count,)
                numericalized.append(n_line)
                borders += [0]
        # remove last, a separator in any case
        borders = borders[:-1]
        return numericalized, borders

    n_hyp, b_hyp = _numericalize_segments(hyp_lines, start_at=0)
    n_ref, b_ref = _numericalize_segments(ref_lines)
    if not len(b_hyp) == len(b_ref):
        print("HYP: ", b_hyp)
        print("REF: ", b_ref)
        raise ValueError("Not the same about of borders %d != %d"
                         % (len(b_hyp), len(b_ref)))

    k = round(1/2*(len(b_hyp)/sum(b_ref))-1)
    window_diff = windowdiff(b_hyp, b_ref, k=k, boundary=1)

    for (cl, cid), (rl, rid) in zip(n_hyp, n_ref):
        if not cl == rl:
            raise ValueError("C99 line does not match reference (hyp: %s, ref: %s), '%s' != '%s'"
                % (hyp, ref, cl, rl))
        print("hyp: %d, ref: %d" % (cid, rid))
        tot_seg += 1
        n_words = len(cl.split())
        tot_words += n_words
        if cid == rid:
            print("\tALIGNED")
            aligned_seg += 1
            aligned_words += n_words

    results = {
        "n_seg": tot_seg,
        "aligned_seg": aligned_seg,
        "r_seg": aligned_seg/tot_seg,
        "n_words": tot_words,
        "aligned_words": aligned_words,
        "r_words": aligned_words/tot_words,
        "windiff": window_diff,
    }
    result_path = hyp.replace(alemi_ext, result_ext)
    print("Writing results to '%s'" % result_path)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def eval_hyp(hyp):
    ref = hyp.replace(alemi_ext, ref_ext)
    seg_n = seg_count(ref)
    _seg_n = seg_count(hyp) + 1  # alemi sources does not start with ===

    if seg_n != _seg_n:
        raise ValueError("Different segment numbers %d != %d for hyp: '%s'"
                         % (seg_n, _seg_n, hyp))
    r = compare(hyp, ref)
    return r


def eval_folder(root_path):
    all_results = {}
    hyps = [
        os.path.join(root_path, f)
        for f in os.listdir(root_path)
        if f.endswith(alemi_ext)
    ]

    sum_keys = ["n_seg", "aligned_seg", "n_words", "aligned_words"]
    total_results = {k: 0 for k in sum_keys}
    weighted_cum_windiff = 0
    for hyp in hyps:
        print("Processing: '%s'" % hyp)
        r = eval_hyp(hyp)
        all_results[hyp] = r

        for k in sum_keys:
            total_results[k] += r[k]

        weighted_cum_windiff += r["windiff"] * r["n_seg"]

    total_results["micro_windiff"] = weighted_cum_windiff / \
        total_results["n_seg"]
    total_results["micro_seg_avg"] = total_results["aligned_seg"] / \
        total_results["n_seg"]
    total_results["micro_word_avg"] = total_results["aligned_words"] / \
        total_results["n_words"]

    all_results_path = os.path.join(root_path, "alemi.all_results.json")
    total_results_path = os.path.join(root_path, "alemi.total_results.json")

    print("Writing all results to '%s'" % all_results_path)
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Writing total results to '%s'" % total_results_path)
    with open(total_results_path, 'w') as f:
        json.dump(total_results, f, indent=2)


if __name__ == "__main__":
        import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root")

    args = parser.parse_args()

    eval_folder(args.root)
