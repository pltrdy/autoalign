import torch
import argparse

from . import word2vec_loader


def sentence_sum_pooling(embeddings):
    return sum(embeddings)
    # return torch.max(torch.stack(embeddings, 1), 1)


def cosine_similarity(s1, s2):
    """
    Args:
        s1: sequence 1 as a list of tensors (word embedding)
        s2: sequence 2 as a list of tensors (word embedding)

    Returns the similarity between sequences, as a 1 element tensor
    """
    v1 = sentence_sum_pooling(s1)
    v2 = sentence_sum_pooling(s2)
    return torch.nn.functional.cosine_similarity(v1, v2, dim=0)


def heuristic_max_similarity(s1, s2):
    """(same doc as `cosine_similarity`
    """
    try:
        stacked_s2 = torch.stack(s2, 0)
    except Exception as e:
        print(s1, s2)
        raise e

    sum_cos = torch.zeros([1])
    for v1 in s1:
        cos_v1_s2 = torch.nn.functional.cosine_similarity(
            v1.expand_as(stacked_s2), stacked_s2)
        sum_cos += cos_v1_s2.max()

    sum_cos.div_(len(s1))
    return sum_cos


SIMILARITY_FCT = {
    'heuristic_max': heuristic_max_similarity,
    'cosine': cosine_similarity
}

SIMILARITY_CHOICES = list(SIMILARITY_FCT.keys()) + ["both"]


class SentenceEmbeddingSimilarity:

    def __init__(self, embedding_path, n_vectors, verbose=False):
        print('Loading embeddings...', end=' ')
        emb_data = word2vec_loader.load_vectors(
            embedding_path, n_vectors, verbose=verbose)
        print('Done.')
        self.embeddings = emb_data['embeddings']
        self.itos = emb_data['itos']
        self.stoi = {v: i for i, v in enumerate(self.itos)}
        if not len(self.itos) == self.embeddings.size(0):
            raise AssertionError

    @property
    def n_dim(self):
        return self.embeddings.size(1)

    @property
    def n_word(self):
        return self.embeddings.size(0)

    def word2emb(self, word):
        """Word (str) to embedding (vector)
        """
        try:
            token = self.stoi[word]
            return self.tok2emb(token)
        except KeyError:
            return

    def tok2emb(self, token):
        """Token (int) to embedding (vector)
        """
        return self.embeddings[token, :].clone()

    def seq2embs(self, seq):
        """Sequence (list of words) to embeddings (list of vectors)
        """
        embs = [self.word2emb(w) for w in seq]
        return [_ for _ in embs if _ is not None]

    def text2embs(self, text):
        """Text (str) to embeddings (list of vectors)
        """
        return self.seq2embs(text.split())

    def sequences_from_file(self, text_path, eos='</s>'):
        with open(text_path, 'rb') as (f):
            text = f.read().decode('utf-8').lower()
        sequences = text.split(eos)
        sequences = [(' ').join(_.split()) for _ in sequences]
        sequences = [_ for _ in sequences if len(_) > 1]
        return sequences

    # def vector_scoring(self, seq0, seq1, similarity_fct='heuristic_max', window_size=500):
    #     if type(similarity_fct) == str:
    #         similarity_fct = SIMILARITY_FCT[similarity_fct]

    #     all_similarities = []

    #     for i, s1 in enumerate(seq1):
    #         window_min = i - (window_size / 2)
    #         window_max = i + (window_size / 2)

    #         for j, s0 in enumerate(seq0):
    #             similarity = 0.0
    #             if s0 is not None and len(s0) > 0:
    #                 if j > window_min and j < window_max:
    #

    def parrallel_matching(self, seqs_ref, seqs_cand,
                           similarity_fct='heuristic_max', output_path=None,
                           window_size=500):
        """

            similarity_fct: str in SIMILARITY_FCT or similarity function
            similarities: [len(seqs_ref) x len(emb_seq_cand)]
        """
        if type(similarity_fct) == str:
            similarity_fct = SIMILARITY_FCT[similarity_fct]
        all_similarities = []
        similarity_fct = heuristic_max_similarity
        print('Loading candidates embeddings...')
        emb_seqs_cand = [self.seq2embs(seq_cand) if len(
            seq_cand) > 2 else None for seq_cand in seqs_cand]
        print('Done.')
        for i, seq_ref in enumerate(seqs_ref):
            print('Processing %d' % i)
            emb_seq_ref = self.seq2embs(seq_ref)
            similarities = []

            window_min = i - (window_size / 2)
            window_max = i + (window_size / 2)
            for j, emb_seq_cand in enumerate(emb_seqs_cand):
                similarity = 0.0
                if emb_seq_cand is not None and len(emb_seq_cand) > 0:
                    if j > window_min and j < window_max:
                        similarity = similarity_fct(emb_seq_ref, emb_seq_cand)
                similarities += [float(similarity)]

            all_similarities += [similarities]

        if output_path is not None:
            print("Saving similarities to: '%s'" % output_path)
            obj = {'similarities': all_similarities, 'sequences': sequences}
            torch.save(obj, output_path)
        return all_similarities

    def texttiling(self, sequences, similarity_fct=heuristic_max_similarity,
                   window_size=5, output_path=None):
        """
        Args:
            sequences: iterator on sequence (may be a file, a list etc...)
        """
        ns = len(sequences)
        wlen = window_size
        window = [self.text2embs(sequences[i]) for i in range(0, wlen + 1)]
        all_similarities = []
        for i in range(ns - 1):
            scur = window[0]
            similarities = []
            for j, snext in enumerate(window[1:]):
                try:
                    similarity = similarity_fct(scur, snext)
                except ValueError as e:
                    print('ValueError when calculating similarity between:')
                    print('#%d[' % i + sequences[i] + ']')
                    print('and')
                    print('#%d[' % (i + j) + sequences[i + j] + ']')
                    raise e

                similarities += [float(similarity)]

            all_similarities += [similarities]
            print('#%d' % i, ('\t').join(['%.2f' % s for s in similarities]))
            window = window[1:]
            next_i = i + wlen
            if next_i < ns:
                window += [self.text2embs(sequences[next_i])]

        if output_path is not None:
            print("Saving similarities to: '%s'" % output_path)
            obj = {'similarities': all_similarities, 'sequences': sequences}
            torch.save(obj, output_path)


def main(word2vec_path, textfile_path, n_vectors,
         window_size, similarity, eos, output_path):
    ses = SentenceEmbeddingSimilarity(word2vec_path, n_vectors)
    sequences = ses.sequences_from_file(textfile_path, eos=eos)
    ses.texttiling(sequences, similarity_fct=SIMILARITY_FCT[similarity], window_size=window_size,
                   output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate sentence similarity')
    parser.add_argument('--word2vec_path', '-w', required=True)
    parser.add_argument('--textfile_path', '-f', required=True)
    parser.add_argument('--n_vectors', '-n', default=150000, type=int,
                        help='Number of vectors to consider (-1 means all)')
    parser.add_argument('--eos', '-eos', default='</s>')
    parser.add_argument('--similarity', '-s',
                        default='heuristic_max', choices=SIMILARITY_FCT.keys())
    parser.add_argument('--output', '-o')
    parser.add_argument('--window', type=int)
    args = parser.parse_args()
    main(args.word2vec_path, args.textfile_path, args.n_vectors,
         args.window, args.similarity, args.eos, args.output)
