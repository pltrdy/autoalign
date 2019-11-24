import os
import autoalign.word2vec_fr.sentence_similarity
here = os.path.abspath(os.path.dirname(__file__))

CBOW_WORD2VEC_NAME = "frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.txt"
CBOW_WORD2VEC_PATH = os.path.join(here, CBOW_WORD2VEC_NAME)

SKIP_WORD2VEC_NAME = "frWac_non_lem_no_postag_no_phrase_500_skip_cut100.txt"
SKIP_WORD2VEC_PATH = os.path.join(here, SKIP_WORD2VEC_NAME)
