from .segmenter import Segmenter

from autoalign.legacy import JobSegmenter


class SimpleSegmenter(Segmenter):
    def __init__(self, *args, tokenizer_properties_name='french', **kwargs):
        super(SimpleSegmenter, self).__init__()

        self.core = JobSegmenter(*args, **kwargs)
        self.core.make_corenlp_client(
            properties_name=tokenizer_properties_name)

    def do_process(self, ctm_paths, docx_path, **_):
        o = {}
        ctm_sen, ctm_sl = self.core.process_ctm(ctm_paths)
        docx_sen, docx_sl = self.core.process_docx(docx_path)

        return ctm_sen, docx_sen, ctm_sl, docx_sl
