import torch
import numpy as np

import autoalign

from autoalign.utils import describe, to_tensor, to_fct_or_none


class ScoreGroup(object):
    def __init__(self, aggregate_fct=None, score_reduce_fct=None):
        self.aggregate_fct = to_fct_or_none(aggregate_fct)
        self.score_reduce_fct = to_fct_or_none(score_reduce_fct)
        pass

    def scatter(self, sentences, **kwargs):
        sentences, group_slices = self.do_scatter(sentences, **kwargs)
        return sentences, group_slices

    def gather(self, scores, ctm_slices, docx_slices, **kwargs):
        return self.do_gather(scores, ctm_slices=ctm_slices,
                              docx_slices=docx_slices, **kwargs)

    def do_scatter(self, sentences, **kwargs):
        raise NotImplementedError()

    def do_gather(self, scores, ctm_slices, docx_slices, **kwargs):
        raise NotImplementedError()


class PaddingGroup(ScoreGroup):
    def __init__(self):
        super(PaddingGroup, self).__init__()

    def do_scatter(self, sentences, **_):
        padded = torch.nn.utils.rnn.pad_sequence(sentences)
        padded.transpose_(0, 1)
        return padded, [slice(i, i+1) for i in range(len(sentences))]

    def do_gather(self, scores, **_):
        return scores


class SlidingWindowGroup2(ScoreGroup):
    def __init__(self, sen_aggregate_fct, score_reduce_fct, initial_gather_score=0.0):
        super(SlidingWindowGroup, self).__init__(
            sen_aggregate_fct, score_reduce_fct)

        self.initial_gather_score = initial_gather_score

    def do_scatter(self, sentences, size, overlap):
        """
        Args:

        Returns:
        """
        n_sen = len(sentences)
        slices = []
        scattered = []
        for i in range(0, n_sen, size - overlap):
            s = slice(i, min(n_sen, i + size))
            slices += [s]

            # print("[%d/%d] in group scorer describe sentences[s], %s" %
            #       (i, n_sen, str(s)))
            # print(describe(sentences[s]))
            try:
                scattered += [self.aggregate_fct(sentences[s])]
            except Exception as e:
                print("[in do_scatter] aggregate error")
                print("Aggregate fct: %s" % self.aggregate_fct.__name__)
                print("Sentences: %s" % str(sentences[s]))
                print("Scattered: %s" % str(scattered))

                raise e
                pass

        print("described scattered: %s " % describe(scattered))
        # stacked_scattered = torch.stack(scattered, 0)
        # return stacked_scattered, slices
        return scattered, slices

    def do_gather(self, scores, ctm_slices, docx_slices):
        """
        Args:
            scores(tensor): [#ctm_slices x #docx_slices]
            ctm_slices(list[slice]): list of scattered ctm slices
            docx_slices(list[slice]): list of scattered docx slices

        Returns:
            f_scores(tensor): [#ctm_sentences x #docx_sentences]
        """

        assert list(scores.size()) == [len(ctm_slices), len(docx_slices)]

        n_ctm = max([_.stop for _ in ctm_slices])
        n_docx = max([_.stop for _ in docx_slices])

        f_scores = torch.zeros([n_ctm, n_docx]) + self.initial_gather_score

        for i in range(scores.size(0)):
            for j in range(scores.size(1)):
                a = f_scores[ctm_slices[i], docx_slices[j]]
                f_scores[ctm_slices[i], docx_slices[j]
                         ] = self.score_reduce_fct([a, scores[i, j]])
                # f_scores[ctm_slices[i], docx_slices[j]] = a * scores[i, j]
        return f_scores


class SlidingWindowGroup(ScoreGroup):
    def __init__(self, sen_aggregate_fct, score_reduce_fct, initial_gather_score=0.0):
        super(SlidingWindowGroup, self).__init__(
            sen_aggregate_fct, score_reduce_fct)

        self.initial_gather_score = initial_gather_score

    def do_scatter(self, sentences, size, overlap):
        """
        Args:

        Returns:
        """
        n_sen = len(sentences)
        slices = []
        scattered = []
        for i in range(0, n_sen, size - overlap):
            s = slice(i, min(n_sen, i + size))
            slices += [s]

            # print("[%d/%d] in group scorer describe sentences[s], %s" %
            #       (i, n_sen, str(s)))
            # print(describe(sentences[s]))
            try:
                scattered += [self.aggregate_fct(sentences[s])]
            except Exception as e:
                print("[in do_scatter] aggregate error")
                print("Aggregate fct: %s" % self.aggregate_fct.__name__)
                print("Sentences: %s" % str(sentences[s]))
                print("Scattered: %s" % str(scattered))

                raise e
                pass

        print("described scattered: %s " % describe(scattered))
        # stacked_scattered = torch.stack(scattered, 0)
        # return stacked_scattered, slices
        return scattered, slices

    def do_gather(self, scores, ctm_slices, docx_slices):
        """
        Args:
            scores(tensor): [#ctm_slices x #docx_slices]
            ctm_slices(list[slice]): list of scattered ctm slices
            docx_slices(list[slice]): list of scattered docx slices

        Returns:
            f_scores(tensor): [#ctm_sentences x #docx_sentences]
        """

        assert list(scores.size()) == [len(ctm_slices), len(docx_slices)]

        n_ctm = max([_.stop for _ in ctm_slices])
        n_docx = max([_.stop for _ in docx_slices])

        f_scores = torch.zeros([n_ctm, n_docx]) + self.initial_gather_score

        for i in range(scores.size(0)):
            for j in range(scores.size(1)):
                a = f_scores[ctm_slices[i], docx_slices[j]]
                f_scores[ctm_slices[i], docx_slices[j]
                         ] = self.score_reduce_fct([a, scores[i, j]])
                # f_scores[ctm_slices[i], docx_slices[j]] = a * scores[i, j]
        return f_scores
