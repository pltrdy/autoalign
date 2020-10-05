# Align then Summarize: Automatic Alignment Methods for Summarization Corpus Creation
## About
This repository contain code of the paper [Align then Summarize: Automatic Alignment Methods for Summarization Corpus Creation](https://arxiv.org/abs/2007.07841)  with instructions for reproduction.

We also provide [`public_meetings`](https://github.com/pltrdy/public_meetings) a novel corpus of meetings (with pairs of transcriptions and reports).

**Cite the paper**: 
```bib
@inproceedings{Tardy2020,
author = {Tardy, Paul and Janiszek, David and Est{\`{e}}ve, Yannick and Nguyen, Vincent},
booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)}, 
keywords = {Alignment,Corpus Annotation,Summarization},
pages = {6718--6724},
title = {{Align then Summarize: Automatic Alignment Methods for Summarization Corpus Creation}},
year = {2020}
}
```
## Getting Started

**1) Clone the repo (along w/ sub-repositories for external models):**   
```bash
git clone https://github.com/pltrdy/autoalign --recursive
```

**2) Install dependencies:**   
```bash
python setup.py install
```

## Automatic Alignment
#### Using parameter sets
The `./experiments` directory contains parameter sets of our experiments.    
We can use specific parameter set (based on their ids) using `./align_mapping.py` (to just align) or `./validate_exp.py` (to align and compare with reference):

So-called `mapping` are just `json` formatted definitions of meetings file paths, e.g.
```
[
    {
        "dir": "/abs/path/to/dir",
        "mapping": [
            {
                "doc": "doc_name.docx"
                "ctm": [
                    "first_ctm_name.ctm",
                    "second_ctm_name.ctm",
                ]
            }, {
                ...
            }
        ]
    }, {
        ...
    }
]

```

You can either specify a `mapping.json` path, or use preset mapping by name e.g. `public_meetings`.

```
out_dir="./exp_embeddings"
mkdir -p "$out_dir"

./align_mapping.py \
    -param_path ./experiments/exp_embeddings_decay.json \
    -mapping_name "public_meetings" \
    -out_dir $out_dir \
    -prefix "public_decay" \
    -n_thread 2 \
    -id 146
```

To evaluate against a set of reference alignment, use the similar `validate_exp.py`:
```
out_dir="./exp_embeddings"
mkdir -p "$out_dir"

./validate_exp.py \
    -path ./experiments/exp_embeddings_decay.json \
    -mapping_name "public_meetings" \
    -root $out_dir \
    -prefix validate_public_decay \
    -aligned_name "public_meetings" \
    -n_thread 2 \
    -ids 146 140

```


#### New experiment
*In short, we define grid-search parameters in `exps.py` and run it by specifying the experiment id to e.g. `./multi_exps.py $id`*


## External models
*External models, namely `TextTiling`, `C99` (+ preliminary work with Alemi's Segmentation) can be found under [`autoalign/external`](https://github.com/pltrdy/autoalign/tree/master/autoalign/external)*
