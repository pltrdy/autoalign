# Align then Summarize: Automatic Alignment Methods for Summarization Corpus Creation
## About
This repository contain code of the paper (under review) with instructions for reproduction.

We also provide [`public_meetings`](https://github.com/pltrdy/public_meetings) a novel corpus of meetings (with pairs of transcriptions and reports).



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
*WIP, tldr, define grid-search parameters in `exps.py` run with `./multi_exps.py`


## External models
WIP
