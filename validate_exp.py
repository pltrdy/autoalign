#!/usr/bin/env python
"""
    The idea is to validate results obtained on single file
    using ./multi_exp.py on multiple files

    Given the experiment params.json files, a list of indices and a
    mapping.json file (check make_aligned_mapping.py)
"""
import os
import json

from multi_exp import run_params
from copy import deepcopy


def validate(params_path, params_id, mapping_path, root="validation", prefix="valid", n_thread=4, aligned_dir="./aligned", force=False, skip_alignment=False, no_compare=False, meta=None, skip_aligned=False, cmp_version=3, one_doc_per_ctm=False):
    """

    Args:
        params_path: path of JSON files with parameters
                     typically from ./multi_exp.py run
        params_id: ids from params to run
        mapping_path: path to mapping json
                      typically from ./make_aligned_mapping.py
        root: output directory
        prefix: output filenames prefix
        aligned_dir: directory to validate on (will process files in this dir)
        force: process even if `root` exists
        skip_alignment: do not run alignment (i.e. only compare)
        no_compare: do not run compare
        meta: data to add to validation info, typically used by
                    ./validate_criterion.py to add metadata
        skip_aligned: whether to ignore existing aignment file or to re-align it

    """
    if cmp_version == 3:
        # Some legacy code involved other comparison processes
        from compare_align import compare_align, INITIAL
    else:
        raise ValueError("Incorrect cmp_version value (%d)" % cmp_version)

    if force and skip_alignment:
        raise ValueError("Cannot set skip_alignment and force together")

    if not os.path.exists(root):
        os.makedirs(root)
    elif not (force or skip_alignment):
        raise ValueError("Root already exists, delete it before")

    valid_info_path = os.path.join(root, "%s_info.json" % prefix)
    with open(valid_info_path, 'w') as f:

        d = {
            "path": params_path,
            "ids": [str(_) for _ in params_id] + [INITIAL],
            "mapping": mapping_path,
            "cmp_version": cmp_version,
        }
        if meta is not None:
            d["meta"] = meta
        json.dump(d, f)

    with open(params_path) as f:
        params = json.load(f)

    with open(mapping_path) as f:
        mapping = json.load(f)

    params = [params[i] for i in params_id]
    # print(json.dumps(params, indent=2))
    # exit()
    results = {}
    for k, v in mapping.items():
        h = k
        aligned_path = os.path.join(aligned_dir, "%s.aligned.pckl" % h)
        doc = v["doc"]
        ctms = v["ctm"]

        _params = deepcopy(params)
        for i in range(len(_params)):
            _params[i]["docx_path"] = doc
            _params[i]["ctm_paths"] = ctms
            if one_doc_per_ctm:
                _params[i]["dp_one_doc_per_ctm"] = [True]

        _prefix = "%s" % (prefix)
        _root = os.path.join(root, h)
        if not os.path.exists(_root):
            os.makedirs(_root)

        if not skip_alignment:
            run_params(_root, _prefix, _params, n_thread=n_thread,
                       skip_aligned=skip_aligned)

        _results = []
        if not no_compare:
            _results = compare_align(
                _root, aligned_path, prefix=_prefix, quiet=False)

            _result_path = os.path.join(_root, "%s.results.json" % _prefix)
            with open(_result_path, "w") as f:
                print("Writing to '%s'" % _result_path)
                json.dump(_results, f, indent=2)
        assert not h in results.keys()
        results[h] = _results

    if not no_compare:
        with open(os.path.join(root, "%s.results.json" % prefix), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-root", type=str)
    parser.add_argument("-prefix", type=str, required=True)
    parser.add_argument("-path", type=str, required=True)
    parser.add_argument("-aligned_dir", default="./aligned")
    parser.add_argument("-ids", nargs="+", type=int, required=True)
    parser.add_argument("-mapping_path", type=str, required=True)
    parser.add_argument("--n_thread", "-n_thread", type=int, default=4)
    parser.add_argument("-force", action="store_true")
    parser.add_argument("-skip_alignment", action="store_true")
    parser.add_argument("-skip_aligned", action="store_true")
    parser.add_argument("-no_compare", action="store_true")
    parser.add_argument("-cmp_v", type=int, default=1)

    args = parser.parse_args()
    validate(args.path, args.ids, args.mapping_path, args.root, args.prefix,
             n_thread=args.n_thread, skip_alignment=args.skip_alignment, force=args.force,
             no_compare=args.no_compare,
             aligned_dir=args.aligned_dir,
             skip_aligned=args.skip_aligned,
             cmp_version=args.cmp_v,
             one_doc_per_ctm=True)
