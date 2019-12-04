#!/usr/bin/env python3
import os
import pickle
import torch
import json
import sys

from nltk.metrics import windowdiff

INITIAL = "<initial>"

results_fields_list = [
    "n_seg", "n_seg_aligned", "ratio_seg_aligned", "win_diff",
    "n_words", "n_words_aligned", "ratio_words_aligned",
    "tot_not_aligned", "words_not_aligned", "ratio_words_not_aligned",
    "alignment", "max_tot_mismatch",
    "hyp_score", "hyp_max_score", "hyp_score_by_seg", "len_path"
]
results_fields = {k: i for i, k in enumerate(results_fields_list)}


def flatten_text(txt):
    return "".join(txt.split())


def aeq(*args):
    if not all([args[0] == _ for _ in args]):
        raise AssertionError(" != ".join(["'%s'" % str(_) for _ in args]))


def first_chars(txt, n):
    return "".join(txt.split())[:n]


def last_chars(txt, n):
    return "".join(txt.split())[-n:]


def _run_compare(all_hyp_ctms, hyp_ctm_aligned, h_n_ctm,
                 h_n_doc, all_final_ctms, dp_table=None, quiet=True):
    forbid_double_ctm = True

    mismatch_count = 0
    tot_mismatch_count = 0
    max_mismatch = 3
    # max_tot_mismatch = 15
    max_tot_mismatch = 30

    def log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    # an alignment notation as list of tuple (align, expected)
    alignment = []

    NOT_ALIGNED = -9
    prev_haligned = NOT_ALIGNED
    prev_faligned = NOT_ALIGNED

    hseg = []
    fseg = []
    fseg_no_neg = ""

    prev_hctm = ""
    hi = 0
    fi = 0

    tot_subsegment = 0

    tot_aligned = 0
    tot_not_aligned = 0
    tot = 0

    tot_words = 0
    words_aligned = 0
    words_not_aligned = 0

    # considering only not -1
    recall_seg = 0
    recall_seg_aligned = 0
    recall_words = 0
    recall_words_aligned = 0

    # number of segments of hypothesis at the end that are not in final
    h_end_cut = 0

    # shifting (
    log("Aligning begining:")
    _hi = hi
    _max_d = 2
    w = 25

    log("hyp shift")
    _max_d2 = 5
    _d2 = 0
    _hyp_shift = False
    while _d2 < _max_d2 and not _hyp_shift:
        _hi = hi + _d2
        _fi = fi + _d2
        _d2 += 1
        while True:
            _ = all_hyp_ctms[_hi][:w]
            log("comparing\n\t'%s'\n\t'%s'" %
                (_[:30], all_final_ctms[fi]['text'][:30]))
            if all_final_ctms[_fi]['text'].startswith(_):
                log("!!! detected hyp_shift")
                _hyp_shift = True
                break
            _hi += 1
            if _hi > _max_d:
                break

    for i in range(_hi):
        alignment += [(hyp_ctm_aligned[i], -2)]
    log("Alignment shifted by %d" % len(alignment))

    if not _hyp_shift:
        debug_n = 150
        print("[Error] content does not match at all")
        print("[Debug] len: final_ctm: %d, hyp_ctms: %d" %
              (len(all_final_ctms), len(all_hyp_ctms)))
        print("[Debug] first %d final: [%s]" % (debug_n, "\n\t".join(
            [_['text'] for _ in all_final_ctms[:debug_n]])))
        print("[Debug] first %d hyp: [%s]" %
              (debug_n, "\n\t".join(all_hyp_ctms[:debug_n])))
        # print("[Debug] " % ())
        # print("[Debug] " % ())
        # print("[Debug] " % ())
        raise ValueError("Content does not match at all")
    hi = _hi
    fi = _fi

    while True:
        f = all_final_ctms[fi]
        log("--- [%d|%d] ---" % (fi, hi))
        while True:
            # skipping double ctm
            # double ctm should not appear anymore
            # however, using `forbid_double_ctm` actually forbids repetitions
            # that may be in the text.
            # also, double ctm check is useless since we chk if hyp == final
            # checking it is enough
            h = all_hyp_ctms[hi]
            hctm = h[:20]

            break

        fctm = f['text']
        h_aligned = int(hyp_ctm_aligned[hi])
        n_words = len(all_hyp_ctms[hi].split())

        h_n_words = len(all_hyp_ctms[hi].split())
        f_n_words = len(fctm.split())

        h_n_chars = len("".join(all_hyp_ctms[hi].split()))
        f_n_chars = len("".join(fctm.split()))

        cmp_n_chars = min(10, min(h_n_chars, f_n_chars))
        cmp_n_words = min(4, min(h_n_words, f_n_words))

        # checking text value
        log("hyp: \t'%s',\nfinal: \t'%s'" % (first_chars(
            all_hyp_ctms[hi], cmp_n_chars), first_chars(fctm, cmp_n_chars)))
        log("\thyp: \t'%s',\n\tfinal: \t'%s'" % (last_chars(
            all_hyp_ctms[hi], cmp_n_chars), last_chars(fctm, cmp_n_chars)))

        def simplify(txt):
            txt = txt.replace("s", "")
            txt = txt.replace("'", "")
            return txt

        if first_chars(all_hyp_ctms[hi], cmp_n_chars) != first_chars(
                fctm, cmp_n_chars):
            # if True:
            if simplify(last_chars(all_hyp_ctms[hi], cmp_n_chars)) != simplify(
                    last_chars(fctm, cmp_n_chars)):
                mismatch_count += 1
                tot_mismatch_count += 1
                print("MISMATCH IN COMPARE ALIGN (%d, %d)" %
                      (mismatch_count, tot_mismatch_count))
                n_words = len(all_hyp_ctms[hi].split())
                if (mismatch_count > max_mismatch or tot_mismatch_count >
                        max_tot_mismatch) and n_words > 2:
                    log("assertion error, logging next hyp/final")
                    log("hyp-1: \n\t'%s', final-1: \n\t'%s'" % (first_chars(
                        all_hyp_ctms[hi - 1], 15), first_chars(all_final_ctms[fi - 1]["text"], 15)))
                    log("hyp: \n\t'%s', final: \n\t'%s'" % (first_chars(
                        all_hyp_ctms[hi], 15), first_chars(all_final_ctms[fi]["text"], 15)))
                    log("hyp+1: \n\t'%s', final+1: \n\t'%s'" % (first_chars(
                        all_hyp_ctms[hi + 1], 15), first_chars(all_final_ctms[fi + 1]["text"], 15)))
                    log("hyp+2: \n\t'%s', final+2: \n\t'%s'" % (first_chars(
                        all_hyp_ctms[hi + 2], 15), first_chars(all_final_ctms[fi + 2]["text"], 15)))
                    raise ValueError()
                else:
                    hi += 1
                    if hi >= len(all_hyp_ctms) - \
                            1 and fi >= len(all_final_ctms) - 1:
                        log("\t~~> Breaking [hi or fi too high] (instead of continuing)")
                        break
                    continue
        mismatch_count = 0

        try:
            f_aligned = int(f['aligned'].split("_")[1])
        except IndexError:
            f_aligned = -1
            tot_not_aligned += 1
            words_not_aligned += n_words

        if f_aligned == h_aligned:
            tot_aligned += 1
            words_aligned += n_words
            log("ALIGNED!!! h[%s]f[%s]" %
                (all_hyp_ctms[hi][:10], fctm[:10]))
        alignment += [(h_aligned, f_aligned)]
        tot_words += n_words
        tot += 1
        log("f-aligned: %d" % f_aligned)
        log("h-aligned: %d" % h_aligned)
        log("%d/%d, %.3f" % (tot_aligned, tot, 100 * tot_aligned / tot))

        if int(f['id'].split('_')[2]) > 0:
            # looking for subsegements i.e. x_y_1, x_y_2 etc..
            while True:
                if fi + 1 >= len(all_final_ctms):
                    break
                f_next = all_final_ctms[fi + 1]
                id_next = f_next['id']
                if int(id_next.split('_')[2]) > 0:
                    tot_subsegment += 1
                    log("using '<%s>'" % id_next)
                    fctm = " ".join([fctm, f_next['text']]).strip()
                    fi += 1
                else:
                    break

        sw = fctm.startswith(hctm)

        # if not sw:
        if (True or hi >= len(all_hyp_ctms) -
                1) and fi >= len(all_final_ctms) - 1:
            # only breaking on final
            log("\t~~> Breaking [hi or fi too high]")

            if hi < len(all_hyp_ctms) - 1:
                h_end_cut = (len(all_hyp_ctms) - 1) - hi

            break

        if hi < len(all_hyp_ctms) - 1:
            log("\t~~> hi += 1")
            hi += 1
        if fi < len(all_final_ctms) - 1:
            fi += 1
            log("\t~~> fi += 1")

        prev_hctm = hctm

        if h_aligned != prev_haligned:
            hseg += [1]
        else:
            hseg += [0]
        if f_aligned != prev_faligned:
            fseg += [1]
        else:
            fseg += [0]
        prev_haligned = h_aligned
        prev_faligned = f_aligned

    log("tot: %d" % tot)
    log("tot_aligned: %d" % tot_aligned)
    log("%%: %.3f" % (100 * tot_aligned / tot))

    # n sentences / n segments
    # Alemi et al 2015
    # (k is taken to be one less than the integer closest to half of the number of elements divided by the number of segments in the reference segmentation.)
    k = round(1 / 2 * (len(hseg) / sum(fseg)) - 1)
    win_diff = windowdiff(hseg, fseg, k=k, boundary=1)

    hyp_score = -1
    hyp_max_score = -1
    hyp_score_by_seg = -1
    len_path = -1
    if dp_table is not None:
        hyp_score = float(dp_table[-1, -1])
        hyp_max_score = float(dp_table.max())
        hyp_score_by_seg = hyp_score / tot

        # thats not trivial, but it remain true...
        len_path = sum(dp_table.size()) - 4

    r = (tot, tot_aligned, 100 * tot_aligned / tot, win_diff,
         tot_words, words_aligned, 100 * words_aligned / tot_words,
         tot_not_aligned, words_not_aligned, 100 * words_not_aligned / tot_words,
         alignment, max_tot_mismatch,
         hyp_score, hyp_max_score, hyp_score_by_seg, len_path)

    # alignment is suppose to be information corresponding to each
    # hypothesis CTM, therefore they may be the same lenght, right?
    real_final_len = len(all_final_ctms) - tot_subsegment - _fi
    matching_hyp_len = len(hyp_ctm_aligned) - tot_mismatch_count - h_end_cut
    real_hyp_len = matching_hyp_len - _hi
    print("\n\n********")
    print("real_final_len: %d (init: %d, subseg: %d, shift: %d)" %
          (real_final_len, len(all_final_ctms), tot_subsegment, _fi))
    print("real_hyp_len: %d (init: %d, mismatch: %d, shift: %d, h_end_cut: %d)" % (
        real_hyp_len, len(hyp_ctm_aligned), tot_mismatch_count, _hi, h_end_cut))
    print("alignment len: %d" % len(alignment))
    print("hseg: %d" % len(hseg))
    print("tot: %d" % (tot))
    assert tot == real_final_len
    assert real_final_len == real_hyp_len
    assert len(alignment) == matching_hyp_len, '%d != %d (%d|%d)' % (
        len(alignment), matching_hyp_len, len(all_final_ctms), tot_subsegment)
    print(alignment)
    return r


def compare_align(hyp_dir, final_path, prefix=None, quiet=False, output=None):
    def log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    # with open(original, 'rb') as f:
    #     orig  = pickle.load(f)

    hyp_paths = sorted([os.path.join(hyp_dir, f)
                        for f in os.listdir(hyp_dir)
                        if f.endswith(".align.pt")
                        and (prefix is None or f.startswith(prefix))],
                       key=lambda x: int(x.split('_')[-1].split('.align.pt')[0]))
    log(hyp_paths)

    with open(final_path, 'rb') as f:
        algn = pickle.load(f)
        final = algn['final']

    f_n_ctm = len(final['ctm'])
    f_n_doc = len(final['doc'])
    all_final_ctms = final['ctm']

    cmp_initial = True

    results = []
    res_dict = {}
    for hyp_path in hyp_paths:
        log("Processing '%s'" % hyp_path)
        hyp = torch.load(hyp_path)

        dp_table = hyp[0].get("dp_table", None)

        h_n_ctm = sum([len(_['ctm']) for _ in hyp])

        h_n_doc = len(hyp)
        all_hyp_ctms = sum([_['ctm'] for _ in hyp], [])

        zda = 5
        log("hyps: %s" % "\n".join([_ for _ in all_hyp_ctms[:zda]]))
        log("\n\n---\nfinals: %s" %
            "\n".join([_["text"] for _ in all_final_ctms[:zda]]))

        hyp_ctm_aligned = [i for i in range(len(hyp))
                           for j in range(len(hyp[i]['ctm']))]
        try:
            r = _run_compare(all_hyp_ctms, hyp_ctm_aligned, h_n_ctm, h_n_doc,
                             all_final_ctms, dp_table=dp_table, quiet=quiet)
            results.append(r)
            res_dict[hyp_path] = r
        except BaseException:
            print("Caught exception with: hyp_path: '%s' final: '%s'" %
                  (hyp_path, final_path), file=sys.stderr)
            raise

    initial = algn['initial']
    h_n_ctm = len(initial['ctm'])
    h_n_doc = len(initial['doc'])
    all_hyp_ctms = [_["text"] for _ in initial["ctm"]]
    hyp_ctm_aligned = [int(_["aligned"].split("_")[1]) for _ in initial["ctm"]]
    try:
        r = _run_compare(all_hyp_ctms, hyp_ctm_aligned, h_n_ctm,
                         h_n_doc, all_final_ctms, quiet=quiet)
    except BaseException:
        print("Exception comparing 'final' and 'initial' of '%s'" % final_path)
        r = (-1, -1, 0, 0,
             -1, -1, 0,
             -1, -1, 0,
             -1, -1,
             -1, -1, -1, -1)
        raise

    results.append(r)
    res_dict[INITIAL] = r

    print(results)

    sorted_i, sorted_results = zip(
        *sorted(enumerate(results), key=lambda x: x[1][2]))
    print("\n".join(["%s : %d %d %.3f (wd: %.3f), words: %d %d %.3f "
                     % (hyp_paths[i] if i < len(hyp_paths) else "initial",
                        sr[0], sr[1], sr[2], sr[3], sr[4], sr[5], sr[6])
                     for i, sr in zip(sorted_i, sorted_results)]))
    print(sorted_results[0][-1])

    if output is not None:
        print("Writing results to '%s'" % output)
        with open(output, 'w') as f:
            json.dump(res_dict, f, indent=2)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_dir", "-hyp", type=str, required=True)
    parser.add_argument("--final", "-f", type=str, required=True)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--prefix")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    compare_align(args.hyp_dir, args.final,
                  output=args.output, quiet=not args.verbose,
                  prefix=args.prefix)
