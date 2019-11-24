import torch
import autoalign
import rouge

from functools import reduce
from autoalign.utils import assert_size, describe, is_fct, to_tensor
from autoalign.legacy.segment_job import CBOW_WORD2VEC_PATH, \
    SKIP_WORD2VEC_PATH, DEFAULT_WORD2VEC_PATH, DEFAULT_N_VECTOR
from autoalign.score.metric import cosine_similarity

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import Normalizer


class Scorer(autoalign.Module):
    def __init__(self):

        super(Scorer, self).__init__()
        self.return_keys = set([
            "scores"
        ])

    def prepare(self, sentences):
        raise NotImplementedError()

    def process(self, ctm_sentences, docx_sentences,
                *args,
                score_group=None,
                scatter_kwargs=None,
                ctm_scatter_kwargs=None, docx_scatter_kwargs=None,
                scorer_prepare_kwargs={},
                score_threshold=None,
                **kwargs):
        if ctm_scatter_kwargs is None and scatter_kwargs is not None:
            ctm_scatter_kwargs = scatter_kwargs

        if docx_scatter_kwargs is None and scatter_kwargs is not None:
            docx_scatter_kwargs = scatter_kwargs

        assert score_group is None or isinstance(score_group, autoalign.score.PaddingGroup) \
            or (ctm_scatter_kwargs is not None
                and docx_scatter_kwargs is not None)
        if docx_scatter_kwargs is None:
            docx_scatter_kwargs = {}
        if ctm_scatter_kwargs is None:
            ctm_scatter_kwargs = {}

        n_ctm, n_docx = len(ctm_sentences), len(docx_sentences)
        ctm_sentences = self.prepare(
            ctm_sentences, with_scores=True, **scorer_prepare_kwargs)
        docx_sentences = self.prepare(docx_sentences, **scorer_prepare_kwargs)
        # raise ValueError()
        print("@Scorer.process after prepare ", describe(ctm_sentences))
        # print("types: %s" % "\n- ".join(["%d %s" % (i, describe(_))
        #                                  for (i, _) in enumerate(ctm_sentences)]))
        if score_group is not None:
            print("before ctm scatter: %d sentences" % len(ctm_sentences))
            ctm_sentences, ctm_group_slices = score_group.scatter(
                ctm_sentences, **ctm_scatter_kwargs)
            print("after ctm scatter: %d sentences, %d slices" %
                  (len(ctm_sentences), len(ctm_group_slices)))
            docx_sentences, docx_group_slices = score_group.scatter(
                docx_sentences, **docx_scatter_kwargs)

            n_ctm_groups = len(ctm_group_slices)
            n_docx_groups = len(docx_group_slices)
            assert len(ctm_sentences) == n_ctm_groups, "%d %d" % (
                len(ctm_sentences), n_ctm_groups)
            assert len(docx_sentences) == n_docx_groups
        # raise ValueError()
        scores = self.do_process(
            ctm_sentences, docx_sentences,
            *args, **kwargs)
        if score_threshold is not None:
            assert len(score_threshold) == 2
            scores_to_replace = scores.lt(score_threshold[0])
            scores[scores_to_replace] = score_threshold[1]
        print(describe(scores.size))
        # print("scores right after processing")
        # print(scores[:5, :5])
        # raise ValueError()
        if score_group is not None:
            # print("scattered scores: %s" % describe(scores))
            assert_size(scores, [n_ctm_groups, n_docx_groups])
            scores = score_group.gather(
                scores, ctm_slices=ctm_group_slices, docx_slices=docx_group_slices)
            # print("scores after gathering")
            # print(scores[:5, :5])
        assert_size(scores, [n_ctm, n_docx])
        return {"scores": scores}


class SentenceEmbeddingScorer(Scorer):
    def __init__(self,
                 word2vec_path=DEFAULT_WORD2VEC_PATH,
                 n_vectors=DEFAULT_N_VECTOR,
                 **scorer_kwags):
        super(SentenceEmbeddingScorer, self).__init__(**scorer_kwags)
        from autoalign.word2vec_fr.sentence_similarity \
            import SentenceEmbeddingSimilarity

        self.ses = SentenceEmbeddingSimilarity(word2vec_path, n_vectors)

    def prepare(self, sequences, pooling_fct='sentence_sum_pooling', min_len=2, stopwords=None, with_scores=False, **_):
        if stopwords == "fr":
            stopwords = autoalign.stopwords.fr
        elif stopwords == "fr_big":
            stopwords = autoalign.stopwords.fr_big
        elif stopwords is not None:
            raise ValueError("No such stopwords '%s'" % stopwords)

        if type(pooling_fct) == str:
            try:
                pooling_fct = getattr(autoalign.score.pooling, pooling_fct)
            except AttributeError:
                raise AttributeError("Unknown pooling function '%s'"
                                     % sentence_sum_pooling)

        embs = []
        for seq in sequences:
            if with_scores:
                seq = [w for w, s in seq]

            if stopwords is not None:
                seq = [_ for _ in seq if _ not in stopwords]

            emb = self.ses.seq2embs(seq)
            if len(emb) == 0:
                emb += [torch.zeros(self.ses.n_dim)]

            emb = pooling_fct(emb) if len(
                emb) > 0 else torch.zeros(self.ses.n_dim)
            if isinstance(emb, torch.Tensor):
                embs += [emb]
            elif type(emb) == list:
                embs += emb

        print("describing embs after SES.prepare %s" % describe(embs))
        return embs

    def do_process(self, ctm_sentences, docx_sentences, metric='cosine_similarity', **kwargs):
        import torch
        if isinstance(ctm_sentences, list):
            ctm_sentences = torch.stack(ctm_sentences, 0)
        if isinstance(docx_sentences, list):
            docx_sentences = torch.stack(docx_sentences, 0)

        # window_size = ses_kwargs.get('window_size', 50)
        # ses_kwargs['window_size'] = window_size

        # print(describe(ctm_sentences))
        # ctm = torch.stack(ctm_sentences, 0)
        # docx = torch.stack(docx_sentences, 0)
        ctm = ctm_sentences
        docx = docx_sentences

        # print(ctm.size())
        # print(docx.size())
        # print("exit @scorer do_process")
        print("metric: %s" % str(metric))
        if not is_fct(metric):
            if type(metric) == str:
                metric = getattr(autoalign.score.metric, metric)
            else:
                raise ValueError("metric must be a function or str")
        scores = metric(ctm, docx)
        # print("scores size: " + str(scores.size()))
        return scores


class FastEmbeddingDTW(SentenceEmbeddingScorer):
    def do_process(self, ctm_sentences, docx_sentences, metric='cosine_similarity', **kwargs):
        import torch
        if isinstance(ctm_sentences, list):
            ctm_sentences = torch.stack(ctm_sentences, 0)
        if isinstance(docx_sentences, list):
            docx_sentences = torch.stack(docx_sentences, 0)

        if not is_fct(metric):
            if type(metric) == str:
                metric = getattr(autoalign.score.metric, metric)
            else:
                raise ValueError("metric must be a function or str")

        import fastdtw
        score, path, scores = fastdtw.fastdtw(
            ctm_sentences, docx_sentences, dist=metric)

        print(type(scores))
        # print(describe(scores))
        s = torch.zeros([len(ctm_sentences), len(docx_sentences)])

        for k, v in scores.items():
            x = k[0]-1
            y = k[1]-1
            s[x, y] = 1 / (v[0] + 1e-8)
        # print(max([k[0] for k in scores.keys()]))
        # print(max([k[1] for k in scores.keys()]))
        # print(len(ctm_sentences))
        # print(len(docx_sentences))

        # print(score)
        # print(describe(path))
        # print(describe(scores))
        # raise ValueError()
        scores = to_tensor(scores)

        return s


class TFIDFScorer(Scorer):
    def __init__(self, **scorer_kwags):
        super(TFIDFScorer, self).__init__(**scorer_kwags)
        # self.tfidf_scoring = scoring_vectors
        self.vectorizer = TfidfVectorizer()
        self.normalizer = Normalizer(copy=False)

    def prepare(self, sentences, *args, lsa_components=10, with_scores=False, **kwargs):
        self.lsa = TruncatedSVD(n_components=lsa_components)

        if type(sentences[0]) == list:
            if with_scores:
                sentences = [" ".join([_[0] for _ in s]) for s in sentences]
            else:
                sentences = [" ".join(s) for s in sentences]

        print("describe sentences in TFIDF prepare %s" % describe(sentences))
        vectors = self.vectorizer.fit_transform(sentences)
        vectors = self.lsa.fit_transform(vectors)
        vectors = self.normalizer.fit_transform(vectors)

        return torch.tensor(vectors).float()

    def do_process(self, ctm_sentences, docx_sentences, **_):
        # scores = self.tfidf_scoring(ctm_sentences, docx_sentences)
        # scores = torch.tensor(scores)
        print("desribe ctm_s in do_process %s" % describe(ctm_sentences))
        print("desribe docx_s in do_process %s" % describe(docx_sentences))

        ctm_sentences = torch.stack(ctm_sentences, 0)
        docx_sentences = torch.stack(docx_sentences, 0)

        scores = cosine_similarity(ctm_sentences, docx_sentences)

        return scores


class ZeroScorer(Scorer):
    def __init__(self, **scorer_kwags):
        super(ZeroScorer, self).__init__(**scorer_kwags)

    def prepare(self, sequences, *_, with_scores=False, **__):
        if with_scores:
            sequences = [[w for w, s in seq] for seq in sequences]
        return sequences

    def do_process(self, ctm_sentences, docx_sentences, **_):
        return torch.zeros([len(ctm_sentences), len(docx_sentences)])


class RougeScorer(Scorer):
    def __init__(self, stats=["f"], all_metrics=False, **scorer_kwags):
        super(RougeScorer, self).__init__(**scorer_kwags)
        self.rouge = rouge.Rouge(stats=stats)
        self.vocab = {}
        self.all_metrics = all_metrics

    def prepare(self, sequences, with_scores=False, **kwargs):
        voc_count = len(self.vocab)
        prepared_sequences = []
        for i, s in enumerate(sequences):
            words = []
            for word in s:
                if with_scores:
                    word = word[0]
                if not(word.startswith("<") and word.endswith(">")):
                    if word in self.vocab.keys():
                        w_id = self.vocab[word]
                    else:
                        w_id = voc_count
                        self.vocab[word] = w_id
                        voc_count += 1
                    words += ["%d" % w_id]
            prepared_sequences.append(" ".join(words))
        return prepared_sequences

    def do_process(self, ctm_sentences, docx_sentences, **_):
        n_ctm = len(ctm_sentences)
        n_docx = len(docx_sentences)
        all_sentences = [
            " " if _ is None else _ for _ in ctm_sentences + docx_sentences]

        scores_ids = list([(i, j) for i in range(n_ctm)
                           for j in range(n_docx)])

        if self.all_metrics:
            sl = [
                rouge.rouge_score.rouge_l_summary_level(
                    [all_sentences[i]], [all_sentences[j]]
                ) for (i, j) in scores_ids
            ]

        s1 = rouge.rouge_score.multi_rouge_n(all_sentences, scores_ids, n=1)
        s2 = rouge.rouge_score.multi_rouge_n(all_sentences, scores_ids, n=2)

        s1 = [_["f"] for _ in s1]
        s2 = [_["f"] for _ in s2]

        if self.all_metrics:
            s = [(_1+_2+_l)/3 for _1, _2, _l in zip(s1, s2, sl)]
        else:
            s = [(_1+_2)/2 for _1, _2 in zip(s1, s2)]

        t = torch.tensor(s).view(n_ctm, n_docx)
        return t
