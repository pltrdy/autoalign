#!/usr/bin/env python
import json
import os
import autoalign


from autoalign.legacy.segment_job import CBOW_WORD2VEC_PATH, SKIP_WORD2VEC_PATH



ctm_paths = [
    "/home/pltrdy/ubiqus_data/wave3/cdata/797_294/audio/797_294_OCIRP.MP3.tdnnf.ctm"
]
docx_path = "/home/pltrdy/ubiqus_data/wave3/cdata/797_294/doc/797294_OCIRP_8eDebatDependance_21012016.docx"


def one_if_zero(x): return x if x != 0.0 else 1.0


def mul(x, y): return x * y


def sum_d5(x, y): return x + (y / 5)


def list_sum(l): return sum([_ for _ in l if _ is not None], [])


exps = [{
    "root": "exp0",
    "params": {
        "segmenter": [autoalign.segment.SimpleSegmenter(interventions=True,
                                                        use_tags=True,
                                                        verbose=False,
                                                        sentence_min_length=2)],
        "scorer": [autoalign.score.SentenceEmbeddingScorer],
        "aligner": [autoalign.align.DPAligner()],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [autoalign.score.SlidingWindowGroup(sum, sum_d5),
                        autoalign.score.SlidingWindowGroup(sum, sum),
                        autoalign.score.PaddingGroup()],
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [5, 10],
            "overlap": [2]
        },
        "scorer_prepare_kwargs": {
            "pooling_fct": ["sentence_sum_pooling", "sentence_max_pooling", "sentence_mean_pooling"]
        }
    },
}, {
    "root": "exp1",
    "params": {
        "segmenter": [autoalign.segment.SimpleSegmenter(interventions=True,
                                                        use_tags=True,
                                                        verbose=True,
                                                        sentence_min_length=2)],
        "scorer": [autoalign.score.SentenceEmbeddingScorer],
        "aligner": [autoalign.align.DPAligner()],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [autoalign.score.SlidingWindowGroup(sum, sum)]
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [2, 3, 5, 7, 10, 15, 20],
            "overlap": [0, 1, 2, 3, 5, 7, 10]
        },
        "scorer_prepare_kwargs": {
            # ,  "sentence_max_pooling", "sentence_mean_pooling"]
            "pooling_fct": ["sentence_sum_pooling"]
        }
    }
}, {
    "root": "exp2",
    "params": {
        "segmenter": [autoalign.segment.SimpleSegmenter(interventions=True,
                                                        use_tags=True,
                                                        verbose=True,
                                                        sentence_min_length=2)],
        "scorer": [autoalign.score.SentenceEmbeddingScorer],
        "aligner": [autoalign.align.DPAligner()],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [autoalign.score.PaddingGroup()]
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [1],
            "overlap": [0]
        },
        "scorer_prepare_kwargs": {
            "pooling_fct": ["sentence_sum_pooling",  "sentence_max_pooling", "sentence_mean_pooling"]
        }
    }
}, {
    "root": "exp3",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": True,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": ["autoalign.score.SentenceEmbeddingScorer"],
        "aligner": ["autoalign.align.DPAligner"],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [("autoalign.score.SlidingWindowGroup", ("sum", "sum"))]
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [2, 3, 4, 5],
            "overlap": [0, 1, 2, 3]
        },
        "scorer_prepare_kwargs": {
            # ,  "sentence_max_pooling", "sentence_mean_pooling"]
            "pooling_fct": ["sentence_sum_pooling"]
        }
    }
}, {
    "root": "exp4",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": False,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": ["autoalign.score.SentenceEmbeddingScorer"],
        "aligner": ["autoalign.align.DPAligner"],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [("autoalign.score.SlidingWindowGroup", ("sum", "sum", 0.0)),
                        ("autoalign.score.SlidingWindowGroup", ("max", "sum", 0.0)),
                        ("autoalign.score.SlidingWindowGroup", ("mean", "sum", 0.0)),
                        ("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0)),
                        ("autoalign.score.SlidingWindowGroup", ("max", "mul", 1.0)),
                        ("autoalign.score.SlidingWindowGroup", ("mean", "mul", 1.0)),
                        ]
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [1, 2, 3, 4, 5, 10],
            "overlap": [0, 1, 2, 3, 5]
        },
        "scorer_prepare_kwargs": {
            # , "sentence_mean_pooling"]
            "pooling_fct": ["sentence_sum_pooling", "sentence_max_pooling"]
        }
    }
}, {  # like exp 4 w/ TFIDFScorer
    "root": "exp5",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": True,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": ["autoalign.score.TFIDFScorer"],
        "aligner": ["autoalign.align.DPAligner"],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [("autoalign.score.SlidingWindowGroup", ("sum", "sum", 0.0)),
                        ("autoalign.score.SlidingWindowGroup", ("max", "sum", 0.0)),
                        ("autoalign.score.SlidingWindowGroup", ("mean", "sum", 0.0)),
                        ("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0)),
                        ("autoalign.score.SlidingWindowGroup", ("max", "mul", 1.0)),
                        ("autoalign.score.SlidingWindowGroup", ("mean", "mul", 1.0)),
                        ]
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [1, 2, 3, 4, 5, 10],
            "overlap": [0, 1, 2, 3, 5]
        },
        "scorer_prepare_kwargs": {
            # , "sentence_mean_pooling"]
            "pooling_fct": ["sentence_sum_pooling", "sentence_max_pooling"]
        }
    }
}, {  # like exp 4 w/ RougeScorer
    "root": "exp6",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": True,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": ["autoalign.score.RougeScorer"],
        "aligner": ["autoalign.align.DPAligner"],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [
            ("autoalign.score.SlidingWindowGroup", ("str_concat", "sum", 0.0)),
            ("autoalign.score.SlidingWindowGroup",
             ("str_concat", "mul", 1.0)),
        ]
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [2, 3, 4, 5, 10],
            "overlap": [0, 1, 2, 3, 5]
        },
        "scorer_prepare_kwargs": {
            # , "sentence_mean_pooling"]
            "pooling_fct": ["sentence_sum_pooling", "sentence_max_pooling"]
        }
    }
}, {  # similar to exp4 to max score
    "root": "exp7",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": True,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": ["autoalign.score.SentenceEmbeddingScorer"],
        "aligner": [
            ("autoalign.align.DPAligner", {"dp_score_scale_fct": "identity"}),
            ("autoalign.align.DPAligner", {"dp_score_scale_fct": "square"}),
            ("autoalign.align.DPAligner", {"dp_score_scale_fct": "pow4"}),
        ],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [
            ("autoalign.score.SlidingWindowGroup", ("sum", "sum", 0.0)),
        ]
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [1],
            "overlap": [0]
        },
        "scorer_prepare_kwargs": {
            # , "sentence_mean_pooling"]
            "pooling_fct": ["sentence_sum_pooling", "sentence_max_pooling"]
        }
    }

}

]

from copy import deepcopy
pow_dpaligner = [
    ("autoalign.align.DPAligner", {"dp_score_scale_fct": "identity"}),
    ("autoalign.align.DPAligner", {"dp_score_scale_fct": "square"}),
    ("autoalign.align.DPAligner", {"dp_score_scale_fct": "pow4"})
]

strconcat_slidingwindow_groups = [
    ("autoalign.score.SlidingWindowGroup", ("str_concat", "sum", 0.0)),
    ("autoalign.score.SlidingWindowGroup", ("str_concat", "mul", 1.0)),
]


exp8 = deepcopy(exps[4])
exp8["root"] = "exp8"
exp8["params"]["aligner"] = deepcopy(pow_dpaligner)
exps.append(exp8)

exp9 = deepcopy(exp8)
exp9["root"] = "exp9"
exp9["params"]["scorer"] = ["autoalign.score.TFIDFScorer"]
exps.append(exp9)

exp10 = deepcopy(exp8)
exp10["root"] = "exp10"
exp10["params"]["scorer"] = ["autoalign.score.RougeScorer"]
exp10["params"]["score_group"] = deepcopy(strconcat_slidingwindow_groups)
exps.append(exp10)
# exps = exps[3:]

exp11 = {
    "root": "exp11",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": True,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": [
            ("autoalign.score.SentenceEmbeddingScorer", {
             "word2vec_path": CBOW_WORD2VEC_PATH, "n_vectors": 50000}),
            ("autoalign.score.SentenceEmbeddingScorer", {
             "word2vec_path": CBOW_WORD2VEC_PATH, "n_vectors": 150000}),
            ("autoalign.score.SentenceEmbeddingScorer", {
             "word2vec_path": CBOW_WORD2VEC_PATH, "n_vectors": 200000}),
            ("autoalign.score.SentenceEmbeddingScorer", {
             "word2vec_path": SKIP_WORD2VEC_PATH, "n_vectors": 50000}),
            ("autoalign.score.SentenceEmbeddingScorer", {
             "word2vec_path": SKIP_WORD2VEC_PATH, "n_vectors": 150000}),
            ("autoalign.score.SentenceEmbeddingScorer", {
             "word2vec_path": SKIP_WORD2VEC_PATH, "n_vectors": 200000}),
        ],
        "aligner": [("autoalign.align.DPAligner", {"dp_score_scale_fct": "pow4"})],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0))],
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [2, 5],
            "overlap": [1, 3]
        },
        "scorer_prepare_kwargs": {
            "pooling_fct": ["sentence_sum_pooling"]
        }
    }
}
exps.append(exp11)


exp12 = deepcopy(exp11)
exp12["root"] = "exp12"
exp12["params"]["scorer"] = [("autoalign.score.SentenceEmbeddingScorer", {
                              "word2vec_path": SKIP_WORD2VEC_PATH, "n_vectors": 150000})]
exp12["params"]["metric"] = ["heuristic_max_similarity"]
exps.append(exp12)


exp13 = deepcopy(exp12)
exp13["root"] = "exp13"
del exp13["params"]["metric"]
exp13["params"]["segmenter"][0][1]["interventions"] = False
exps.append(exp13)


exp14 = deepcopy(exp8)
exp14["root"] = "exp14"
exp14["params"]["scorer"] = [("autoalign.score.FastEmbeddingDTW", {
                              "word2vec_path": SKIP_WORD2VEC_PATH, "n_vectors": 150000})]
exp14["params"]["metric"] = ["cosine_similarity"]
exps.append(exp14)


souffle_segmenter = [("autoalign.segment.SimpleSegmenter",
                      {
                          "interventions": True,
                          "use_tags": True,
                          "verbose": True,
                          "sentence_min_length": 2
                      })
                     ]
exp15 = deepcopy(exp12)
exp15["root"] = "exp15"
exp15["params"]["metric"] = ["cosine_similarity"]
exp15["params"]["segmenter"] = deepcopy(souffle_segmenter)
exps.append(exp15)


# based on exp_8_299 explore some dp dhw dgw variations
dpaligner_dgw_dhw = [
    ("autoalign.align.DPAligner", {"dp_score_scale_fct": "pow4"}),
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4", "dp_dhw": 1, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4",              "dp_dgw": 0.9}),
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4", "dp_dhw": 0.9}),
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4", "dp_dhw": 0.9, "dp_dgw": 0.9}),
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4", "dp_dhw": 1}),
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4",                "dp_dgw": 1}),
]
exp16 = {
    "root": "exp16",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": False,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": [
            ("autoalign.score.SentenceEmbeddingScorer", {
             "word2vec_path": SKIP_WORD2VEC_PATH, "n_vectors": 150000}),
        ],
        "aligner": dpaligner_dgw_dhw,
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0))],
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [2, 5],
            "overlap": [1, 3]
        },
        "scorer_prepare_kwargs": {
            "pooling_fct": ["sentence_sum_pooling"]
        }
    }
}
exps.append(exp16)


# Exploring dhw, dgw w/ square pre-scale
exp17 = deepcopy(exp16)
exp17["root"] = "exp17"
for aligner_conf in exp17["params"]["aligner"]:
    aligner_conf[1]["dp_score_scale_fct"] = "square"
exps.append(exp17)


# Exploring dhw, dgw w/ identity pre-scale
exp18 = deepcopy(exp16)
exp18["root"] = "exp18"
for aligner_conf in exp18["params"]["aligner"]:
    aligner_conf[1]["dp_score_scale_fct"] = "identity"
exps.append(exp18)


# Exploring exp8 without dgw/dhw
dpaligner_no_dgw_dhw = [
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "identity", "dp_dhw": 1, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "square", "dp_dhw": 1, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "pow4", "dp_dhw": 1, "dp_dgw": 1}),
]

exp19 = deepcopy(exp8)
exp19["root"] = "exp19"
exp19["params"]["aligner"] = dpaligner_no_dgw_dhw
exps.append(exp19)

# Experimenting with pre-DP score scale i.e. softmax
# long story short: it suxx
exp20 = {
    "root": "exp20",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": True,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": [
            ("autoalign.score.SentenceEmbeddingScorer", {
             "word2vec_path": SKIP_WORD2VEC_PATH, "n_vectors": 150000}),
        ],
        "aligner": [
            "autoalign.align.DPAligner",
        ],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0))],
        'dp_score_scale_fct': ["identity", "square", "pow4"],
        "dp_score_pre_scale_fct": ["identity", "softmax"],
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [2, 5],
            "overlap": [1, 3]
        },
        "scorer_prepare_kwargs": {
            "pooling_fct": ["sentence_sum_pooling"]
        }
    }
}
exps.append(exp20)


# exp8_299 explore segmentation tweaks
exp21 = deepcopy(exp20)
exp21["root"] = "exp21"
del exp21["params"]["dp_score_pre_scale_fct"]
exp21["params"]["scpres"] = ["autoalign.score.SentenceEmbeddingScorer"]
exp21["params"]["dp_score_scale_fct"] = ["pow4"]
exp21["sub_params"] = {"scatter_kwargs": {"size": [5], "overlap": [3]},
                       "scorer_prepare_kwargs": {"pooling_fct": ["sentence_sum_pooling"]}}
exp21["params"]["segmenter"] = [
    ("autoalign.segment.SimpleSegmenter", {
     "interventions": True, "use_tags": True, "verbose": False, "sentence_min_length": 1}),
    ("autoalign.segment.SimpleSegmenter", {
     "interventions": True, "use_tags": True, "verbose": False, "sentence_min_length": 2}),
    ("autoalign.segment.SimpleSegmenter", {
     "interventions": True, "use_tags": True, "verbose": False, "sentence_min_length": 3}),
    ("autoalign.segment.SimpleSegmenter", {
     "interventions": True, "use_tags": True, "verbose": False, "sentence_min_length": 4}),
    ("autoalign.segment.SimpleSegmenter", {"interventions": True, "use_tags": True,
                                           "verbose": False, "sentence_min_length": 1, "sentence_slices": True}),
    ("autoalign.segment.SimpleSegmenter", {"interventions": True, "use_tags": True,
                                           "verbose": False, "sentence_min_length": 2, "sentence_slices": True}),
    ("autoalign.segment.SimpleSegmenter", {"interventions": True, "use_tags": True,
                                           "verbose": False, "sentence_min_length": 3, "sentence_slices": True}),
    ("autoalign.segment.SimpleSegmenter", {"interventions": True, "use_tags": True,
                                           "verbose": False, "sentence_min_length": 4, "sentence_slices": True}),
]
exps.append(exp21)

exp22 = deepcopy(exp20)
exp22["root"] = "exp22"
del exp22["params"]["dp_score_pre_scale_fct"]
exp22["params"]["scpres"] = ["autoalign.score.SentenceEmbeddingScorer"]
exp22["params"]["dp_score_scale_fct"] = ["pow4"]
exp22["sub_params"] = {"scatter_kwargs": {"size": [5], "overlap": [3]},
                       "scorer_prepare_kwargs": {"pooling_fct": ["sentence_sum_pooling"]}}
exp22["params"]["segmenter"] = [
    ("autoalign.segment.SimpleSegmenter", {
     "interventions": True, "use_tags": True, "verbose": False, "sentence_min_length": 2}),
]
exp22["params"]["dp_one_doc_per_ctm"] = [True, False]
exps.append(exp22)

exp23 = deepcopy(exp8)
exp23["root"] = "exp23"
exp23["params"]["score_group"] = [
    ("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0))
]
exp23["sub_params"]["scorer_prepare_kwargs"] = {
    "pooling_fct": ["sentence_sum_pooling"]}
exp23["params"]["dp_one_doc_per_ctm"] = [True]
exps.append(exp23)

# 22 w/ f36f8581848a8c67bddd63d007dbc70e [not anymore]
exp24 = deepcopy(exp22)
exp24["root"] = "exp24"
exps.append(exp24)

exp25 = deepcopy(exp22)
exp25["root"] = "exp25"
exps.append(exp25)

# re-run 19 w/ less params
# useless grid over pooling and agg (fixing to sum)
# red mul is >= sum
exp26 = deepcopy(exp19)
exp26["params"]["score_group"] = [
    ("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0))
]
exp26["root"] = "exp26"
exp26["sub_params"]["scorer_prepare_kwargs"] = {
    "pooling_fct": ["sentence_sum_pooling"]}
exps.append(exp26)

# exp27 = DTW w/o DGW/DHW
exp27 = deepcopy(exp14)
exp27["root"] = "exp27"
exp27["params"]["aligner"] = dpaligner_no_dgw_dhw
exps.append(exp27)

# exp28 = TFIDF w/o dgw/dhw
exp28 = deepcopy(exp9)
exp28["root"] = "exp28"
exp28["params"]["aligner"] = dpaligner_no_dgw_dhw
exps.append(exp27)

# exp 29 = exploring no pooling i.e. word level alignment
exp29 = deepcopy(exp8)
exp29["root"] = "exp29"
exp29["sub_params"]["scorer_prepare_kwargs"] = {
    "pooling_fct": ["no_pooling"],
    "stopwords": ["fr"]
}
exp29["params"]["score_group"] = [
    ("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0))
]
exps.append(exp29)

# exp30 = shallow exploration dhw/dgw with square and identity
exp30 = deepcopy(exp16)
exp30["root"] = "exp30"
exp30["sub_params"]["scatter_kwargs"] = {
    "size": [5],
    "overlap": [3]
}
id_dpaligner_dgw_dhw = deepcopy(dpaligner_dgw_dhw)
sq_dpaligner_dgw_dhw = deepcopy(dpaligner_dgw_dhw)
for i in range(len(id_dpaligner_dgw_dhw)):
    id_dpaligner_dgw_dhw[i][1]["dp_score_scale_fct"] = "identity"
    sq_dpaligner_dgw_dhw[i][1]["dp_score_scale_fct"] = "square"

exp30["params"]["aligner"] = id_dpaligner_dgw_dhw + sq_dpaligner_dgw_dhw

exps.append(exp30)

# exp31 = better exp8
#   - adding dgw/dhw exploration
#   - removing sentence pooling
#   - only considering sum/mul agg/red
exp31 = deepcopy(exp8)
exp31["root"] = "exp31"
exp31["params"]["align"] = [
    ("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0))
]
exp31["sub_params"]["scorer_prepare_kwargs"] = {
    "pooling_fct": ["sentence_sum_pooling"],
}
exp31["params"]["aligner"] = dpaligner_dgw_dhw + \
    id_dpaligner_dgw_dhw + sq_dpaligner_dgw_dhw
exps.append(exp31)

# exp32: experimenting threshold in ~ top exp8 setup
exp32 = {
    "root": "exp32",
    "params": {
        "segmenter": [
            ("autoalign.segment.SimpleSegmenter",
             {"interventions": True,
              "use_tags": True,
              "verbose": False,
              "sentence_min_length": 2})
        ],
        "scorer": ["autoalign.score.SentenceEmbeddingScorer"],
        "aligner": [
            ("autoalign.align.DPAligner", {"dp_score_scale_fct": "pow4"}),
        ],
        "ctm_paths": [["/home/pltrdy/ubiqus_data/wave3/cdata/797_294/audio/797_294_OCIRP.MP3.tdnnf.ctm"]],
        "docx_path": ["/home/pltrdy/ubiqus_data/wave3/cdata/797_294/doc/797294_OCIRP_8eDebatDependance_21012016.docx"],
        "score_group": [
            ("autoalign.score.SlidingWindowGroup", ["sum", "mul", 1.0]),
        ],
        "score_threshold": [
            [0.0, 0.0],
            [0.7, 0.001],
            [0.5, 0.001],
            [0.25, 0.001],
            [0.1, 0.001],

            [0.7, 0.1],
            [0.5, 0.1],
            [0.25, 0.1],
            [0.1, 0.1],

            [0.7, 0.2],
            [0.5, 0.2],
            [0.25, 0.2],
            [0.1, 0.2],

        ],
        "dp_one_doc_per_ctm": [True],
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [5],
            "overlap": [3]
        },
    }
}
exps.append(exp32)

# exp33: naive
exp33 = {
    "root": "exp33",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": True,
                           "sentence_min_length": 2
                       })
                      ],
        "scorer": ["autoalign.score.ZeroScorer"],
        "aligner": ["autoalign.align.DiagonalAligner"],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [None],
    },
    "sub_params": {}
}
exp33["params"]["dp_one_doc_per_ctm"] = [True]
exps.append(exp33)

# exp34 exp8 w/ dp_one_doc_per_ctm
exp34 = deepcopy(exp8)
exp34["root"] = "exp34"
exp34["params"]["dp_one_doc_per_ctm"] = [True]
exp34["sub_params"]["scorer_prepare_kwargs"]["pooling_fct"] = [
    "sentence_sum_pooling"]
exp34["params"]["score_group"] = [
    ('autoalign.score.SlidingWindowGroup', ('sum', 'mul', 1.0))]
exps.append(exp34)

# exp35: extending exp23 [best exp8 w/ one_ctm_per_doc] w/ dgw/dhw
exp35 = deepcopy(exp23)
exp35["root"] = "exp35"
exp35["params"]["aligner"] = id_dpaligner_dgw_dhw + \
    sq_dpaligner_dgw_dhw + dpaligner_dgw_dhw
exps.append(exp35)


# exp36: == exp23 with more agg/red
exp36 = deepcopy(exp8)
exp36["root"] = "exp36"
# exp36["params"]["score_group"] = [
#     ("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0))
# ]
exp36["sub_params"]["scorer_prepare_kwargs"] = {
    "pooling_fct": ["sentence_sum_pooling"]}
exp36["params"]["dp_one_doc_per_ctm"] = [True]
exps.append(exp36)


# exp37: exp35 w/ lower dhw
dpaligner_dhw99_pow4 = [
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4", "dp_dhw": 0.99, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4", "dp_dhw": 0.99}),

    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4", "dp_dhw": 0.999, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
        "dp_score_scale_fct": "pow4", "dp_dhw": 0.999}),
]
dpaligner_dhw99_square = deepcopy(dpaligner_dhw99_pow4)
for i in range(len(dpaligner_dhw99_square)):
    dpaligner_dhw99_square[i][1]["dp_score_scale_fct"] = "square"


exp37 = deepcopy(exp35)
exp37["root"] = "exp37"
exp37["params"]["aligner"] = dpaligner_dhw99_square + dpaligner_dhw99_pow4
exps.append(exp37)

# exp38 - it's best exp 35 w/ some dhw exploration
exp38 = {
    "root": "exp38",
    "params": {
        "segmenter": [("autoalign.segment.SimpleSegmenter",
                       {
                           "interventions": True,
                           "use_tags": True,
                           "verbose": False,
                           "sentence_min_length": 2,
                           "tokenizer_properties_name": "french"
                       })
                      ],
        "scorer": ["autoalign.score.SentenceEmbeddingScorer"],
        "aligner": [
        ],
        "ctm_paths": [ctm_paths],
        "docx_path": [docx_path],
        "score_group": [
            ("autoalign.score.SlidingWindowGroup", ("sum", "mul", 1.0)),
        ],
        "dp_one_doc_per_ctm": [True]
    },
    "sub_params": {
        "scatter_kwargs": {
            "size": [2],
            "overlap": [1]
        },
        "scorer_prepare_kwargs": {
            "pooling_fct": ["sentence_sum_pooling"]
        }
    }
}
aligned_more_dhw_dgw = []
# for dhw 0.90 is bad, 1 is bad, 0.9999 is ok, explore ]0.90, 0.9999]
# for dgw 0.90 is bad, 1 is best, 0.9999 is good
for dhw in [0.995, 0.9990, 0.9995, 0.99995, 0.99999]:
    for dgw in [None, 0.999, 0.99999, 1]:
        for scale in ["pow4", "square"]:
            kwargs = {"dp_score_scale_fct": scale, "dp_dhw": dhw}
            if dgw is not None:
                kwargs["dp_dgw"] = dgw
            t = ("autoalign.align.DPAligner", kwargs,)
            aligned_more_dhw_dgw.append(t)
exp38["params"]["aligner"] = aligned_more_dhw_dgw

exps.append(exp38)


# exp39: 38 w/ legacy tokenizer
exp39 = deepcopy(exp38)
exp39['root'] = "./exp39"
exp39['params']['segmenter'][0][1]['tokenizer_properties_name'] = 'legacy'
exps.append(exp39)

# exp40: 39 w/ dhw=0.9999 (attempt to reproduce 35)
aligned_reproduce_dhw_dgw = []
# for dhw 0.90 is bad, 1 is bad, 0.9999 is ok, explore ]0.90, 0.9999]
# for dgw 0.90 is bad, 1 is best, 0.9999 is good

exp40 = deepcopy(exp39)
exp40['root'] = "exp40"
for dhw in [None, 0.9999]:
    for dgw in [None, 0.999, 0.99999, 1]:
        for scale in ["pow4", "square"]:
            kwargs = {"dp_score_scale_fct": scale}
            if dhw is not None:
                kwargs["dp_dhw"] = dhw
            if dgw is not None:
                kwargs["dp_dgw"] = dgw
            t = ("autoalign.align.DPAligner", kwargs,)
            aligned_reproduce_dhw_dgw.append(t)
exp40["params"]["aligner"] = aligned_reproduce_dhw_dgw
exps.append(exp40)


# exp41: exp8 w/ dp_one_doc_per_ctm; same in idea than 34 but we suspect it to have failed
exp8_best_aligner = []
for dgw in [0.9999, 1]:
    for dhw in [0.9999, 1]:
        exp8_best_aligner += [
            ("autoalign.align.DPAligner",
                {
                    "dp_score_scale_fct": "pow4",
                    "dp_dhw": dhw,
                    "dp_dgw": dgw
                }
             )
        ]
exp41 = deepcopy(exp8)
exp41["root"] = "exp41"
exp41["params"]["dp_one_doc_per_ctm"] = [True]
exp41["sub_params"]["scorer_prepare_kwargs"]["pooling_fct"] = [
    "sentence_sum_pooling"]
exp41["params"]["score_group"] = [
    ('autoalign.score.SlidingWindowGroup', ('sum', 'mul', 1.0))]
exp41["params"]["aligner"] = exp8_best_aligner
exp41["sub_params"]["scatter_kwargs"] = {"size": [5], "overlap": [3]}
exps.append(exp41)

# exp42: exp8, just to check what dgw/dhw = n/a,n/a means
#      : exp41 w/o dp_one_doc_per_ctm
exp42 = deepcopy(exp41)
exp42["root"] = "exp42"
exp42["params"]["dp_one_doc_per_ctm"] = [False]
exps.append(exp42)

# exp43: rlly reproducing 35
#        40 w/ dgw=1, dhw
aligned_rlly_reproduce_dhw_dgw = []

exp43 = deepcopy(exp39)
exp43['root'] = "exp43"
for dhw in [None, 0.9999, 1]:
    for dgw in [None, 0.999, 0.9999, 0.99999, 1]:
        for scale in ["pow4", "square"]:
            kwargs = {"dp_score_scale_fct": scale}
            if dhw is not None:
                kwargs["dp_dhw"] = dhw
            if dgw is not None:
                kwargs["dp_dgw"] = dgw
            t = ("autoalign.align.DPAligner", kwargs,)
            aligned_rlly_reproduce_dhw_dgw.append(t)
exp43["params"]["aligner"] = aligned_rlly_reproduce_dhw_dgw
exp43["params"]['segmenter'][0][1]['tokenizer_properties_name'] = 'french'
exp43["params"]["dp_one_doc_per_ctm"] = [True]
exps.append(exp43)

# exp44 = 42 w/ french tok
exp44 = deepcopy(exp42)
exp44["root"] = "exp44"
exp44["params"]['segmenter'][0][1]['tokenizer_properties_name'] = 'french'
exps.append(exp44)

# 45: 43 (which is roughly 35) w/ ubi embeddings
exp45 = deepcopy(exp43)
exp45["root"] = "exp45"
exp45["params"]["scorer"] = [
    ("autoalign.score.SentenceEmbeddingScorer",
     {"word2vec_path": "/home/pltrdy/autoalign/ubi_vectors.txt", "n_vectors": 149996}
     ),
]
exps.append(exp45)

# 46: 45 w/ stopwords
exp46 = deepcopy(exp45)
exp46["root"] = "exp46"
exp46["sub_params"]["scorer_prepare_kwargs"]["stopwords"] = ["fr_big"]
exps.append(exp46)

# 47: 35 w/ exploring pooling
exp47 = deepcopy(exp35)
exp47["root"] = "exp47"
exp47["sub_params"]["scorer_prepare_kwargs"]["pooling_fct"] = [
    "sentence_sum_pooling",  "sentence_max_pooling", "sentence_mean_pooling"]
exps.append(exp47)

# 48: 35 w/ stopwords (note 46 uses pre-trained embeddings)
exp48 = deepcopy(exp35)
exp48["root"] = "exp48"
exp48["sub_params"]["scorer_prepare_kwargs"]["stopwords"] = ["fr_big"]
exps.append(exp48)


# 49: 35 w/ TF-IDF stopwords (or not), and various pooling
exp49 = deepcopy(exp35)
exp49["root"] = "exp49"
exp49["params"]['segmenter'][0][1]['tokenizer_properties_name'] = 'french'
exp49["sub_params"]["scorer_prepare_kwargs"]["stopwords"] = ["fr_big", None]
exp49["sub_params"]["scorer_prepare_kwargs"]["pooling_fct"] = [
    "sentence_sum_pooling",  "sentence_max_pooling", "sentence_mean_pooling"]
exp49["params"]["scorer"] = ["autoalign.score.TFIDFScorer"]
exps.append(exp49)

# 50: 49 w/ ROUGE
exp50 = deepcopy(exp49)
exp50["root"] = "exp50"
exp50["params"]["scorer"] = ["autoalign.score.RougeScorer"]
del exp50["sub_params"]["scorer_prepare_kwargs"]["pooling_fct"]
exp50["params"]["score_group"] = deepcopy(strconcat_slidingwindow_groups)
exps.append(exp50)

# 51: 49 w/ DTW
exp51 = deepcopy(exp49)
exp51["root"] = "exp51"
exp51["params"]["scorer"] = ["autoalign.score.FastEmbeddingDTW"]
exps.append(exp51)


# exp52: bestof 35 with max/mean pooling
exp52 = deepcopy(exp35)
exp52["root"] = "exp52"
exp52["sub_params"]["scorer_prepare_kwargs"]["pooling_fct"] = [
    "sentence_max_pooling", "sentence_mean_pooling"]
exp52["params"]['segmenter'][0][1]['tokenizer_properties_name'] = 'french'
exp52["params"]["dp_one_doc_per_ctm"] = [True]
exp52["sub_params"]["scatter_kwargs"] = {
    "size": [1, 2, 5],
    "overlap": [0, 1, 3]
}
exp52["params"]["aligner"] = [
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "pow4", "dp_dhw": 1, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "pow4", "dp_dhw": 0.9999, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "pow4", "dp_dhw": 0.9999, "dp_dgw": 0.9999}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "pow4", "dp_dhw": 1, "dp_dgw": 0.9999}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "square", "dp_dhw": 1, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "square", "dp_dhw": 0.9999, "dp_dgw": 1}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "square", "dp_dhw": 0.9999, "dp_dgw": 0.9999}),
    ("autoalign.align.DPAligner", {
     "dp_score_scale_fct": "square", "dp_dhw": 1, "dp_dgw": 0.9999}),
]
exps.append(exp52)

# 53: 52 W/ dtw
#
exp53 = deepcopy(exp52)
exp53["root"] = "exp53"
exp53["scorer"] = ["autoalign.score.FastEmbeddingDTW"]
exps.append(exp53)

# 54: ROUGE, with R1, R2, RL
# Not working, RL would be too long
exp54 = deepcopy(exp10)
exp54["root"] = "exp54"
exp54["params"]['segmenter'][0][1]['tokenizer_properties_name'] = 'french'
exp54["params"]["scorer"] = [
    ("autoalign.score.RougeScorer", {"all_metrics": True}),
]
exps.append(exp54)

# 55: tfidf without dgw/dhw
exp55 = deepcopy(exp9)
exp55["root"] = "exp55"
exp55["params"]['segmenter'][0][1]['tokenizer_properties_name'] = 'french'
for i, aligner in enumerate(exp55["params"]["aligner"]):
    d = aligner[1]
    d["dp_dgw"] = 1.0
    d["dp_dhw"] = 1.0
    exp55["params"]["aligner"][i] = (aligner[0], d,)
exps.append(exp55)

# 56 ROUGE without dgw/dhw
exp56 = deepcopy(exp10)
exp56["root"] = "exp55"
exp56["params"]['segmenter'][0][1]['tokenizer_properties_name'] = 'french'
for i, aligner in enumerate(exp56["params"]["aligner"]):
    d = aligner[1]
    d["dp_dgw"] = 1.0
    d["dp_dhw"] = 1.0
    exp56["params"]["aligner"][i] = (aligner[0], d,)
exps.append(exp56)

# 57: top tf-idf
