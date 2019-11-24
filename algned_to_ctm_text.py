#!/usr/bin/env python
import pickle


def algned_to_ctm_text(algned_path, out_path):
    positive_only = False
    with open(algned_path, 'rb') as f:
        algned = pickle.load(f)

    final_ctms = algned["final"]["ctm"]
    out = open(out_path, "w")
    ref_path = out_path+".seg.ref"
    out_ref = open(ref_path, "w")

    prev_aligned = None
    if not positive_only:
        print("==========", file=out_ref)

    for ctm in final_ctms:
        if ctm["aligned"] == '' or ctm["aligned"] == '-1' or ctm["aligned"] == -1:
            if positive_only:
                continue
        if prev_aligned is not None and ctm["aligned"] != prev_aligned:
            print(ctm["aligned"], prev_aligned)
            print("==========", file=out_ref)
        print(ctm["text"], file=out)
        print(ctm["text"], file=out_ref)
        prev_aligned = ctm["aligned"]
    print("==========", file=out_ref)
    print("Wrote to: '%s'" % out_path)
    print("Wrote to: '%s'" % ref_path)
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("algned_path")
    parser.add_argument("out_path")

    args = parser.parse_args()

    algned_to_ctm_text(args.algned_path, args.out_path)
