#!/usr/bin/env python3
import json
import argparse

from validate_exp import validate
from compare_align import INITIAL


def validate_criterion(cmp_file, params_path, mapping_path, gt=None, root="validation", prefix="valid", n_thread=4, force=False, skip_alignment=False, skip_aligned=False, cmp_version=1, aligned_dir="./aligned", one_doc_per_ctm=False):
    criterion = None

    if gt is not None:
        def criterion(x): return x > gt

    if criterion is None:
        raise ValueError("No parameters to set 'criterion'")

    with open(cmp_file) as f:
        cmp_data = json.load(f)

    ids = []
    for k, v in cmp_data.items():
        acc = v[2]

        if criterion(acc):
            exp_id = k.split(".align.pt")[0].split("_")[-1]
            if exp_id == INITIAL:
                continue
            exp_id = int(exp_id)
            ids += [exp_id]
    print("#%d exps: %s" % (len(ids), ", ".join([str(_) for _ in ids])))

    meta = {"cmp_file": cmp_file, "gt": gt, "prefix": prefix}
    validate(params_path, ids, mapping_path, root=root,
             prefix=prefix, n_thread=n_thread, force=force, skip_alignment=skip_alignment,
             skip_aligned=skip_aligned, meta=meta, cmp_version=cmp_version,
             aligned_dir=aligned_dir,
             one_doc_per_ctm=one_doc_per_ctm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-root", type=str)
    parser.add_argument("-prefix", type=str)
    parser.add_argument("-params_path", type=str, required=True)
    parser.add_argument("-mapping_path", type=str, required=True)
    parser.add_argument("-n_thread", "-t", type=int, default=4)
    parser.add_argument("-force", action="store_true")
    parser.add_argument("-skip_alignment", action="store_true")
    parser.add_argument("-skip_aligned", action="store_true")
    parser.add_argument("-aligned_dir", default="./aligned")
    parser.add_argument("-no_compare", action="store_true")

    parser.add_argument("-cmp_path", "-f", type=str, required=True)
    parser.add_argument("-gt", type=float)
    parser.add_argument("-cmp_version", "-cmp_v", type=int, default=1)
    parser.add_argument("-one_doc_per_ctm", action="store_true")

    args = parser.parse_args()
    validate_criterion(args.cmp_path, args.params_path, args.mapping_path, args.gt,
                       args.root, args.prefix, args.n_thread,
                       skip_alignment=args.skip_alignment,
                       skip_aligned=args.skip_aligned,
                       force=args.force,
                       cmp_version=args.cmp_version,
                       aligned_dir=args.aligned_dir,
                       one_doc_per_ctm=args.one_doc_per_ctm)
