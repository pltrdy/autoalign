#!/usr/bin/env python
import os
import json
import autoalign

from copy import deepcopy
from multi_exp import run_params


def align_mapping(prefix, mapping, param_path, param_id, root_dir, out_dir,
                  skip_aligned=True, subfolder=False, viz_dp=False, n_thread=8):

    with open(param_path) as f:
        params = json.load(f)

    params = params[param_id]
    print(params)

    all_params = []
    for h, pair in mapping:
        job_dir = job["dir"]
        job_root = os.path.join(root_dir, job_dir)

        for pair in job["mapping"]:
            if pair["doc"] == "NOMAPPING":
                continue
            if len(pair["ctm"]) == 0:
                continue

            p = deepcopy(params)

            audio_sub = "audio" if subfolder else ""
            doc_sub = "doc" if subfolder else ""
            ctm_paths = [os.path.join(job_root, audio_sub, c)
                         for c in pair["ctm"]]
            docx_path = os.path.join(job_root, doc_sub, pair["doc"])
            for path in ctm_paths + [docx_path]:
                assert os.path.exists(path), "'%s' does not exist" % path

            p['docx_path'] = docx_path
            p['ctm_paths'] = ctm_paths
            p['name'] = os.path.basename(p['docx_path'])
            p['align_slices_verbose'] = True
            if not viz_dp:
                p['dp_output_path'] = None
            all_params += [p]
    run_params(out_dir, prefix, all_params,
               n_thread=n_thread, skip_aligned=skip_aligned)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", required=True)
    parser.add_argument("-mapping_name")
    parser.add_argument("-mapping_path")
    parser.add_argument("-out_dir", "-o", required=True)
    parser.add_argument("-param_path", required=True)
    parser.add_argument("-param_id", "-id",
                        default=146, type=int)
    parser.add_argument("-root_dir", default="")
    parser.add_argument("-no_skip_aligned", action="store_true")
    parser.add_argument("-subfolder", action="store_true")
    parser.add_argument("-viz_dp", action="store_true")
    parser.add_argument("-n_thread", type=int, default=8)
    args = parser.parse_args()

    mapping = autoalign.mapping.load_mapping_args(args)

    align_mapping(args.prefix, mapping, args.param_path, args.param_id, args.root_dir, args.out_dir,
                  skip_aligned=not args.no_skip_aligned, subfolder=args.subfolder, viz_dp=args.viz_dp, n_thread=args.n_thread)
