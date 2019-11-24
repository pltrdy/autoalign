#!/bin/bash
set -e

# from http://fauconnier.github.io/
url="http://embeddings.org/frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.bin"
wget $url

# skip
# http://embeddings.org/frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin
