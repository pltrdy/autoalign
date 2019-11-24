#!/usr/bin/env python
import pickle
import json
import os
import autoalign

# from multiprocessing import Pool
# pathos is less restricting during the serialization
from pathos.multiprocessing import ProcessingPool as Pool

from autoalign.pipeline import instantiate
from grid_search import grid_search_params, subgrid
from autoalign.legacy.dynamic_prog_similarities import viz_alignement
from autoalign.legacy.segment_job import CBOW_WORD2VEC_PATH, SKIP_WORD2VEC_PATH
from json import JSONEncoder

from autoalign.utils.parallel import parallel_map
from copy import deepcopy
from exps import exps


def prepare_exp(params, sub_grid):
    to_instantiate = ['segmenter', 'scorer', 'aligner', 'score_group']

    from copy import deepcopy
    iparams = deepcopy(params)
    for ik in to_instantiate:
        iparams[ik] = instantiate(iparams[ik])

    def _prep(_params, _sub_grid):
        _all_params = subgrid(_params, _sub_grid)
        _grid = grid_search_params(_all_params)

        _grid = [_ for _ in _grid
                 if (not "scatter_kwargs" in _.keys()
                     or _["scatter_kwargs"]["size"] - _["scatter_kwargs"]["overlap"] > 0)]
        return _grid

    grid = _prep(params, sub_grid)
    igrid = _prep(iparams, sub_grid)
    return igrid, grid


def run_exp_args(args):
    run_exp(*args)


def run_exp(root_dir, p, prefix=None, skip_aligned=False):
    modules_key = ['segmenter', 'scorer', 'aligner']
    modules = []

    name = p['name']
    del p['name']

    _prefix = name
    if prefix is not None:
        _prefix = "%s_%s" % (prefix, _prefix)

    output_html = os.path.join(root_dir, "%s.align.html" % _prefix)
    output_pt = os.path.join(root_dir, "%s.align.pt" % _prefix)
    if os.path.exists(output_pt) and skip_aligned:
        print("Skipping (not overwriting) '%s'" % output_pt)
        return

    for mk in modules_key:
        modules += [p[mk]]
        del p[mk]

    pipeline = autoalign.Pipeline(*modules)
    o, a = pipeline(**p)

    extra_data = {k: a[k] for k in ["dp_table", "dp_path"]}
    viz_alignement(a['scores'], o["alignment"], a['docx_slices'], a['ctm_slices'], a['docx_sentences'],
                   a['ctm_sentences'], output_html=output_html, output_pt=output_pt, extra_data=extra_data)


def run_exps(exp_ids, n_thread=1, skip_aligned=False):
    for i in exp_ids:
        exp = exps[i]
        root = exp['root']
        igrid, grid = prepare_exp(exp['params'], exp['sub_params'])

        if not os.path.exists(root):
            os.makedirs(root)

        prefix = "exp_%d" % i
        for i, (p, ip) in enumerate(zip(grid, igrid)):
            name = "%s_%d" % (prefix, i)
            p['name'] = name
            ip['name'] = name

        params_path = os.path.join(root, "%s_params.json" % prefix)

        with open(params_path, 'w') as f:
            json.dump(grid, f, indent=2)

        params_grid = igrid
        print("n configs: %d" % len(params_grid))
        _prefix = None

        parallel = "pathos_fixed"
        if parallel == "pathos":
            with Pool(processes=n_thread) as pool:
                pool.map(run_exp_args, [
                         (root, p, _prefix, skip_aligned) for p in params_grid])
        elif parallel == "pathos_fixed":
            parallel_map(run_exp_args, [
                (root, p, _prefix, skip_aligned) for p in params_grid],
                n_thread)
        else:
            for p in params_grid:
                run_exp_args([root, p, _prefix, skip_aligned])


def run_params(root, prefix, params, n_thread=1, skip_aligned=True):
    print("n configs: %d" % len(params))

    to_instantiate = ['segmenter', 'scorer', 'aligner', 'score_group']
    instantiated_params = {k: [] for k in to_instantiate}
    instantiated_objects = {k: [] for k in to_instantiate}

    iparams = deepcopy(params)

    # instantiating while avoiding duplicates
    for ik in to_instantiate:
        for i, p in enumerate(params):
            v = params[i][ik]

            # checking if exist
            name = params[i]['name']
            _prefix = name
            if prefix is not None:
                _prefix = "%s_%s" % (prefix, _prefix)

            # output_html = os.path.join(root, "%s.align.html" % _prefix)
            output_pt = os.path.join(root, "%s.align.pt" % _prefix)
            if os.path.exists(output_pt) and skip_aligned:
                print("Skipping (not overwriting) '%s'" % output_pt)
                continue

            try:
                inst_id = instantiated_params[ik].index(v)
                inst = instantiated_objects[ik][inst_id]
                print("Found repetitive key %s" % ik)
            except ValueError:
                try:
                    if params[i][ik] is None:
                        print("Skip None param i: %d, key: %s" % (i, ik))
                        continue
                    inst = instantiate([params[i][ik]])
                except ValueError as e:
                    print("Cannot instantiate %s" % str(params[i][ik]))
                    raise e
                try:
                    assert type(inst) == list
                    assert len(inst) == 1, "%d != 1" % len(inst)
                except AssertionError:
                    print(params)
                    print("[ERROR] AssertionError on params: %d, %s" % (i, ik))
                    print(params[i][ik])
                    raise
                inst = inst[0]

                instantiated_params[ik] += [params[i][ik]]
                instantiated_objects[ik] += [inst]
                print("Instantiating key %s" % ik)

            iparams[i][ik] = inst
    try:
        parallel_map(run_exp_args, [
            (root, p, prefix, skip_aligned) for p in iparams],
            n_thread)
    except:
        raise
    #  with Pool(processes=n_thread) as pool:
    #      pool.map(run_exp_args, [(root, p, prefix, skip_aligned)
    #                              for p in iparams])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_ids", type=int, nargs='+')
    parser.add_argument("--n_thread", "-n_thread", "-t", type=int, default=4)
    args = parser.parse_args()

    run_exps(args.exp_ids, args.n_thread)
