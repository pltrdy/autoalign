import autoalign

from autoalign.utils import describe


class Segmenter(autoalign.Module):
    def __init__(self):
        super(Segmenter, self).__init__()
        self.return_keys = set([
            "ctm_sentences",
            "docx_sentences",
            "ctm_slices",
            "docx_slices",
        ])
        # TODO make 3rd party segmenter

    def process(self, ctm_paths, docx_path, **kwargs):
        ctm_sen, docx_sen, ctm_sl, docx_sl = self.do_process(
            ctm_paths, docx_path, **kwargs)
        # print(describe(ctm_sen))
        # print(describe(docx_sen))
        #
        # print(ctm_sl)
        # raise ValueError()
        return {
            "ctm_sentences": ctm_sen,
            "docx_sentences": docx_sen,
            "ctm_slices": ctm_sl,
            "docx_slices": docx_sl
        }
