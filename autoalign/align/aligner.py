import autoalign

from autoalign.utils import assert_size, to_fct
from autoalign.legacy.dynamic_prog_similarities import dp, analyse_dp, \
    align_slices, replace_nan, viz_dp


class Aligner(autoalign.Module):
    def __init__(self):
        super(Aligner, self).__init__()
        self.return_keys = set([
            "alignment"
        ])

    def process(self, scores, ctm_slices, docx_slices, **kwargs):
        n_ctm = max([_.stop for _ in ctm_slices])
        n_docx = max([_.stop for _ in docx_slices])
        assert_size(scores, [n_ctm, n_docx])

        o = self.do_process(scores, ctm_slices, docx_slices, **kwargs)
        if type(o) == tuple:
            assert len(o) == 2
            assert type(o[1]) == dict
            d = {"alignment": o[0]}
            for k, v in o[1].items():
                d[k] = v
            return d
        else:
            return {"alignment": o}


class DiagonalAligner(Aligner):
    def __init__(self, **_):
        super(DiagonalAligner, self).__init__()

    def do_process(self, scores, ctm_slices, docx_slices, align_sentence_level=True,
                   dp_one_doc_per_ctm=False, **_):
        len_ctm, len_doc = list(scores.size())
        ratio = len_ctm / len_doc
        cur_c = 0
        cur_d = 0

        path = [(cur_d, cur_c)]
        while cur_c < len_ctm-1 or cur_d < len_doc-1:
            scores[cur_c, cur_d] = 1

            cur_ratio = cur_c if cur_d == 0 else cur_c / cur_d
            if (cur_ratio < ratio or cur_d == len_doc-1) and cur_c + 1 < len_ctm:
                cur_c += 1
                print("cur_c %d %d" % (cur_c, len_ctm))
            elif cur_d+1 < len_doc:
                cur_d += 1
                print("cur_d: %d %d" % (cur_d, len_doc))
            else:
                print("should leave cur_c: %d/%d, cur_d: %d/%d" %
                      (cur_c, len_ctm, cur_d, len_doc))

            path += [(cur_d, cur_c)]
        # raise ValueError("DiagAlign done: cur_c: %d/%d, cur_d: %d/%d" % (cur_c, len_ctm, cur_d, len_doc))
        print("Slices:")
        print(ctm_slices)
        print("Path:")
        print(path)
        aligned_slices = align_slices(path[::-1],
                                      docx_slices,
                                      ctm_slices,
                                      one_doc_per_ctm=dp_one_doc_per_ctm,
                                      verbose=True)

        return aligned_slices, {"dp_table": scores, "dp_path": path}


class DPAligner(Aligner):
    def __init__(self, dp_score_scale_fct=lambda x: x, dp_dgw=0.9999, dp_dhw=0.9999,):
        super(DPAligner, self).__init__()
        self.dp_score_scale_fct = to_fct(dp_score_scale_fct)
        self.dp_dhw = dp_dhw
        self.dp_dgw = dp_dgw

    def do_process(self, scores, ctm_slices, docx_slices, dp_output_path=None,
                   dp_dgw=None, dp_dhw=None, dp_score_scale_fct=None,
                   dp_score_pre_scale_fct=lambda x: x,
                   dp_one_doc_per_ctm=False,
                   align_slices_verbose=False,
                   **_):
        """
        Args:
            scores(Tensor): [#ctm x #docx] 
        """
        # raise ValueError()
        # dp_* kwargs can be set from both constructor and do_process
        if dp_dgw is None:
            dp_dgw = self.dp_dgw
        if dp_dhw is None:
            dp_dhw = self.dp_dhw
        if dp_score_scale_fct is None:
            dp_score_scale_fct = self.dp_score_scale_fct
        dp_score_scale_fct = to_fct(dp_score_scale_fct)

        dp_score_pre_scale_fct = to_fct(dp_score_pre_scale_fct)
        scores = replace_nan(scores, 0)

        table, hist = dp(scores,
                         score_scale_fct=dp_score_scale_fct,
                         dgw=dp_dgw,
                         dhw=dp_dhw,
                         score_pre_scale_fct=dp_score_pre_scale_fct)
        x_count, dp_path = analyse_dp(hist)

        # raise ValueError()
        aligned_slices = align_slices(dp_path,
                                      docx_slices,
                                      ctm_slices,
                                      verbose=align_slices_verbose,
                                      one_doc_per_ctm=dp_one_doc_per_ctm)
        print(aligned_slices)
        if dp_one_doc_per_ctm:
            assert len(aligned_slices) == len(docx_slices)
            assert len(sum(aligned_slices, [])) == len(ctm_slices), "%d %d" % (
                len(sum(aligned_slices, [])), len(ctm_slices))

        if dp_output_path is not None:
            viz_dp(table, scores, table, hist, docx_slices, ctm_slices, dp_path,
                   output_path=dp_output_path)

        print(scores[:5, :5])
        print(table[:5, :5])
        return aligned_slices, {"dp_table": table, "dp_path": dp_path}
