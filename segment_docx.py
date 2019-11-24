#!/usr/bin/env python
from autoalign.legacy.segment_job import JobSegmenter
from pathos.multiprocessing import ProcessingPool as Pool


def segment_docx_list(list_path, print_path=False):
    n_thread = 4
    with open(list_path, 'r') as f:
        paths = [_.strip() for _ in f]

    job_segmenter = JobSegmenter(use_tags=True, interventions=True,
                                 sentence_min_length=2)
    job_segmenter.make_corenlp_client()

    def _process_path(path):
        sentences, slices = job_segmenter.process_docx(path)
        if print_path:
            print(("\n%s: " % path).join([" ".join(_) for _ in sentences]))
        else:
            print("\n".join([" ".join(_) for _ in sentences]))

    with Pool(processes=n_thread) as pool:
        pool.map(_process_path, paths)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-list", "-l", required=True, type=str)
    parser.add_argument("-print_path", action="store_true")
    args = parser.parse_args()

    segment_docx_list(args.list, args.print_path)
