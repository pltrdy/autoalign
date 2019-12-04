#!/usr/bin/env python
"""
    LEGACY UTILITIES:
        still used, wrapped in other classes

    Good example:

    Need to export export CORENLP_HOME=/$HOME/stanford_corenlp/

    We use the following 'notation'
        level           := int

        word            := string
        sentence        := [ $word ]
        paragraph       := [ $sentence ]
        sentences       := [ $sentence ]

        section         := {
                                "level"     : $level,
                                "content"   : $paragraph,
                                "childs"    : [ $section ]
                            }
        structure       := $section w/ level=-1,content=[],childs=[ $section ]

        flat_structure  := [ $paragraph ]
        flat_section    := [ $sentence ]
            (== paragraph == sentences)
"""
import argparse
import docx
import time
import urllib3
import requests
import stanfordnlp.server.client as corenlp_client

import autoalign
import torch
import sys
import os

from stanfordnlp.server import CoreNLPClient
from autoalign.docx_utils import docx_iter, elmt2txt, is_p_elmt, is_tbl_elmt, is_row_elmt

DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULT_HEADING_IDX = 10
CBOW_WORD2VEC_PATH = autoalign.word2vec_fr.CBOW_WORD2VEC_PATH
SKIP_WORD2VEC_PATH = autoalign.word2vec_fr.SKIP_WORD2VEC_PATH
DEFAULT_WORD2VEC_PATH = SKIP_WORD2VEC_PATH

DEFAULT_N_VECTOR = 150000

TAGLESS = "tagless"
TAG_H = "h"
TAG_P = "p"
TAG_NAME = "nom"
TAG_TABLE = "table"
TAG_ROW = "row"


def otag(tag):
    return "<%s>" % tag


def ctag(tag):
    return "</%s>" % tag


def is_ctag(word):
    return word.startswith("</") and word.endswith(">")


def is_otag(word):
    return not is_ctag and word.startswith("<") and word.endswith(">")


def tabs(idx):
    return '\t' * idx


def flatten_list(_list):
    return sum(_list, [])


class JobSegmenter(object):
    """
        default_heading_idx(int): level associated with default header
                                  headers are usually h1 -> h6 therefore any
                                  integer > 6 may be ok.
        verbose(bool): well...
        use_tags(bool): whether to tag data (with paragraphs, title, names)
        interventions(bool): switch to intervention segmentation
                             i.e. map paragraphs to intervention in ctm
    """

    def __init__(self,
                 default_heading_idx=DEFAULT_HEADING_IDX,
                 verbose=False,
                 use_tags=False,
                 interventions=False,
                 sentence_slices=False,
                 tfidf=False,
                 sentence_min_length=1):
        #         corenlp_port=9000):
        self.corenlp_client = None
        self.ses = None
        self.default_heading_idx = default_heading_idx
        self.n_headings = 6
        self.paragraph_min_word_length = 2
        self.verbose = verbose
        self.use_tags = use_tags
        self.interventions = interventions
        self.sentence_slices = sentence_slices
        self.tfidf = tfidf
        self.sentence_min_length = sentence_min_length
        # self.corenlp_port = corenlp_port

        # if self.interventions:
        #     raise ValueError("Intervention mode not implemented")

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.corenlp_client:
            self.corenlp_client.stop()

    def make_sentence_similarity(self, word2vec_path=DEFAULT_WORD2VEC_PATH,
                                 n_vectors=DEFAULT_N_VECTOR):
        if self.tfidf:
            from exp_tfidf_lsa_kmeans.tfidf import scoring
            self.tfidf_scoring = scoring
        else:
            from word2vec_fr.sentence_similarity import SentenceEmbeddingSimilarity
            self.ses = SentenceEmbeddingSimilarity(word2vec_path, n_vectors)

    def make_corenlp_client(self, annotators=["tokenize", "ssplit"],
                            endpoint="http://localhost:9000",
                            properties_name="french",
                            properties_dict=None,
                            quiet=True):
        LEGACY_PROPERTIES = {}
        FRENCH_PROPERTIES = {
            "tokenize.language": "French",
            "tokenize.options": "ptb3Dashes=true"
        }
        PROPERTIES = {
            "legacy": LEGACY_PROPERTIES,
            "french": FRENCH_PROPERTIES
        }
        if properties_dict is not None:
            properties = properties_dict
        else:
            if properties_name in PROPERTIES.keys():
                properties = PROPERTIES[properties_name]
            else:
                raise ValueError("Unknow properties '%s'" % properties_name)

        devnull = open(os.devnull)
        stdout = devnull if quiet else sys.stdout
        stderr = devnull if quiet else sys.stderr
        self.corenlp_client = \
            CoreNLPClient(annotators=annotators,
                          endpoint=endpoint,
                          stdout=stdout,
                          stderr=stderr,
                          memory="8G",
                          heapsize="8G",
                          threads=8,
                          timeout=15000,
                          properties=properties
                          )

    def heading_idx(self, style_name):
        if style_name is None:
            return self.default_heading_idx

        if style_name.lower().startswith(
                "heading") or style_name.lower().startswith("titre"):
            try:
                idx = int(style_name[-1]) - 1
                return idx
            except ValueError as e:
                return self.default_heading_idx
        else:
            return self.default_heading_idx

    def is_toc(self, style_name):
        if style_name is None:
            return False
        style_name = style_name.lower()
        return (style_name.lower().startswith("contents")
                or style_name.lower().startswith("toc")
                or style_name.lower().startswith("en-t")
                or style_name.lower().startswith("tm"))

    def is_name(self, style_name):
        if style_name is None:
            return False
        style_name = style_name.lower()
        return (style_name.startswith('nom')
                or style_name.startswith('intervenant'))

    def section(self, level=None, content=[], parent=None):
        if level is None:
            level = self.self.default_heading_idx
        kwargs = locals()
        self = kwargs.pop("self")
        parent = kwargs.pop('parent')
        d = dict(kwargs)
        d["childs"] = []
        if parent is not None:
            d["parent"] = parent
            parent["childs"].append(d)
        return d

    def annotate(self, text):
        """
        Args:
            text(string)
        Returns:
            annotation object
        """
        if self.corenlp_client is None:
            raise ValueError("'self.corenlp_client' is None. "
                             "Use 'make_corenlp_client' before calling "
                             "'annotate'")
        while True:
            try:
                r = self.corenlp_client.annotate(text)
                break
            except (requests.exceptions.ConnectionError, corenlp_client.PermanentlyFailedException,
                    urllib3.exceptions.MaxRetryError):
                print("too many requests, sleeping")
                time.sleep(0.75)
        return r

    def get_sentences(self, words, lower=True, no_minimum=False,
                      with_scores=False, debug=False):
        # return [_.split() for _ in text.split(".")]
        return self.corenlp_get_sentences(words, lower=lower, no_minimum=no_minimum,
                                          with_scores=with_scores, debug=debug)

    def corenlp_get_sentences(self, words, lower=True,
                              no_minimum=False, with_scores=False, debug=False):
        """
        Args:
            sentences: list[list[word]] if not with_scores
                       list[list[ [word; score]] otherwise
        Returns:
            sentences: list of sentences (list of word (string)
                list[list[str]]
                or words is [word, score] if with_scores
        """
        _debug = debug

        def debug(*args, **kwargs):
            if _debug:
                print(*args, **kwargs)

        def maybe_lower(t):
            return t.lower() if lower else t

        if not with_scores:
            # note replacing spe quote only needed for french for aujourd'hui
            ann = self.annotate(" ".join(words).replace("’", "'"))

            sentences = [[maybe_lower(token.word)
                          for token in sentence.token]
                         for sentence in ann.sentence]
            sentences = [s for s in sentences
                         if no_minimum or len(s) >= self.sentence_min_length]
            return sentences
        else:
            words, scores = zip(*[(w, s) for w, s in words])
            debug(words)
            debug(scores)
            text = " ".join(words)
            ann = self.annotate(text.replace("’", "'"))

            count = 0
            sentences = []
            prev_word = ""
            prev_score = ""
            word_done = True
            wip = ""
            for sentence in ann.sentence:
                sentences.append([])
                for token in sentence.token:
                    debug("'%s' ~= '%s'" % (words[count], token.word))
                    wip += token.word
                    score = scores[count]
                    if not wip == words[count]:
                        debug("Incomplete word, wip='%s'" % wip)
                    else:
                        wip = ""
                        count += 1
                    # if not token.word == words[count]:
                    #     if word_done and words[count].startswith(token.word):
                    #         word_done = False
                    #         debug("Incomplete word: begining")
                    #     elif words[count].endswith(token.word):
                    #         debug("Incomplete word: end")
                    #         word_done = True
                    #         count += 1
                    #     elif token.word in words[count]:
                    #         debug("Incomplete word: middle")
                    #         pass
                    #     else:
                    #         raise ValueError("mismatch '%s' and '%s'" % (words[count], token.word))
                    # else:
                    #     count += 1
                    sentences[-1].append([maybe_lower(token.word), score])

            def to_old_style(s): return " ".join(
                [w[0] for w, _ in s]).replace("' ", "'").split()
            sentences = [s for s in sentences
                         if no_minimum or len(to_old_style(s)) >= self.sentence_min_length]

            assert count == len(scores) == len(words)
            return sentences

    def flatten_document(self, document, implicit_nom=False, exclude_toc=True):
        """

        Args:
            document(docx.Document)

        Returns:
            sections(list[section]) with:
                section: list[sentence]
                sentence: list[word(str)]
                finally sections is list[list[list[word(str)]]]

                NOTE: now word is actually [word, score]

        """
        implicit_nom_file = open('implicit_nom.lst_', 'a')
        unique_noms = set()

        cur_section = []
        sections = [cur_section]
        cur_lvl = -1
        style_error_p = []
        cur_txt_len = 0

        last_tag = TAGLESS

        # for p in document.paragraphs:
        for elmt in docx_iter(document):
            tag = TAGLESS

            if is_p_elmt(elmt):
                self.log("p elmt, style=%s" % elmt.style)
                try:
                    style = elmt.style
                    if exclude_toc and self.is_toc(style):
                        continue

                    lvl = self.heading_idx(style)

                    if lvl == self.default_heading_idx:
                        tag = TAG_P
                    else:
                        tag = TAG_H

                    if self.is_name(style):
                        tag = TAG_NAME

                except AttributeError as e:
                    style_error_p += [elmt]
                    lvl = self.default_heading_idx
                    tag = TAG_P
                    raise e
            else:
                if is_tbl_elmt(elmt):
                    tag = TAG_TABLE
                elif is_row_elmt(elmt):
                    tag = TAG_ROW

                lvl = self.default_heading_idx

            self.log("lvl: %d" % lvl)

            # sentences = list[list[str]]
            # words = list[str]
            words = elmt2txt(elmt).split()
            self.log("'%s'\n" % [_.lower() for _ in words[:150]])

            no_minimum = True
            sentences = self.get_sentences(
                words, no_minimum=no_minimum, with_scores=False)
            if implicit_nom:
                assert self.interventions

                def _add_implicit_nom(nom):
                    __nom = " ".join(nom)
                    if not __nom in unique_noms:
                        print(__nom, file=implicit_nom_file)
                        unique_noms.add(__nom)

                if len(sentences) > 0:
                    s = " ".join(sentences[0])

                    if "--" in sentences[0]:
                        # <nom> -- intervention
                        count = sentences[0].count("--")
                        pos = sentences[0].index("--")

                        if count == 1 and pos < 7 and len(sentences) > 1:
                            nom = s.split("--")[0].split()
                            intervention = " ".join(s.split("--")[1:]).split()
                            _nom = ["<nom>"] + nom + ["</nom>"]
                            _intervention = ["<%s>" % tag] + intervention
                            sentences[-1].append("</%s>" % tag)
                            _sentences = [_nom, _intervention] + sentences[1:]
                            cur_section = _sentences
                            sections.append(cur_section)

                            _add_implicit_nom(nom)
                            continue
                    elif len(sentences[0]) < 5 and any([s.lower().startswith(_)
                                                        for _ in ["monsieur", "madame",
                                                                  "m.", "mme.", "mr."]]):
                        # <monsieur|madame|..> <nom> \n text
                        # print("monsieur|madame detected")
                        nom = sentences[0]
                        _nom = ["<nom>"] + nom + ["</nom>"]
                        _sentences = [_nom]
                        if len(sentences) > 1:
                            sentences[1] = ["<%s>" % tag] + sentences[1]
                            sentences[-1] = sentences[-1] + ["</%s>" % tag]
                            _sentences += sentences[1:]

                        cur_section = _sentences
                        sections.append(cur_section)

                        _add_implicit_nom(nom)
                        continue

            words = flatten_list(sentences)
            cur_txt_len += len(words)

            if self.use_tags:
                if not len(sentences) > 0:
                    sentences = [[]]
                    # continue

                sentences[0] = ["<%s>" % tag] + sentences[0]
                sentences[-1] = sentences[-1] + ["</%s>" % tag]
            if "".join(words) == "e-customer":
                words = ["e", "-", "customer"]

            if len(words) == 0:
                continue
            if len(words) < self.paragraph_min_word_length:
                if not tag in [TAG_NAME, TAG_H]:
                    continue
            if self.interventions:
                if tag in [TAG_NAME, TAG_H] and last_tag != TAG_H:
                    cur_section = sentences
                    sections += [cur_section]
                else:
                    cur_section += sentences
            else:
                # sections mode:
                if lvl <= cur_lvl:
                    # new section
                    if cur_txt_len > 0:
                        cur_section = sentences
                        sections += [cur_section]
                        cur_txt_len = 0
                    cur_lvl = lvl
                else:
                    # appending content to current section
                    cur_section += sentences

                    if lvl < self.n_headings:
                        # only updates the level in case of header
                        cur_lvl = lvl
            last_tag = tag
        return sections

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def process_docx(self, docx_path, implicit_nom=False, verbose=False):
        """
        Args:
            docx_path(str)

        Returns:
            sentences(list[str])
            slices(list[slice])
        """
        document = docx.Document(docx_path)
        # structure = self.get_docx_structure(document)
        # flat_structure = self.flatten_section(structure)
        sections = self.flatten_document(document, implicit_nom=implicit_nom)

        sentences = flatten_list(sections)
        if self.sentence_slices and not self.interventions:
            slices = [slice(i, i + 1) for i in range(len(sentences))]
        else:
            slices = []
            lower = 0
            for section in sections:
                if len(section) == 0:
                    continue
                upper = lower + len(section)
                slices += [slice(lower, upper)]
                lower = upper

        return sentences, slices

    def process_ctm(self, ctm_paths, get_scores=False, debug=False):
        """
        Args:
            ctm_paths(list[string])

        Returns:
            ctm_slices(list[slice])
            ctm_sentences(list[string])
        """
        paroles = []
        cur_sentence = ""
        ctm_sentences = []
        # 1. CTM ->> List of paroles
        for ctm_path in ctm_paths:
            with open(ctm_path, 'rb') as f_ctm:
                for line in f_ctm:
                    try:
                        word, score = line.decode('utf-8').split("\t")[4:6]
                        score = float(score)
                    except UnicodeDecodeError as e:
                        print("UnicodeDecodeError on file '%s'" % ctm_path)
                        raise e

                    if word.startswith("<start="):
                        if len(paroles) == 0 or len(paroles[-1]) > 0:
                            paroles.append([])
                        continue
                    paroles[-1].append([word, score])

        # 2. List of paroles ->> List of lists of sentences (and ranges)
        if self.sentence_slices:
            assert not get_scores, "Not implemented"
            ctm_sentences = []
            for parole in paroles:
                sentences = [_ for _ in self.get_sentences(
                    parole, with_scores=True) if len(_) > 0]
                ctm_sentences += sentences

            parole_slices = [slice(i, i + 1)
                             for i in range(len(ctm_sentences))]
        else:
            ctm_sentences = []
            parole_slices = []
            lower = 0
            for i_parole in range(len(paroles)):
                parole = paroles[i_parole]
                self.log("***")
                self.log(parole)
                ## ann = client.annotate(parole)
                # sentences = [[token.word.lower()
                # for token in sentence.token]
                # for sentence in ann.sentence]
                sentences = self.get_sentences(
                    parole, with_scores=True, debug=debug)

                if len(sentences) == 0:
                    continue
                self.log(sentences)
                ctm_sentences += sentences
                upper = lower + len(sentences)
                parole_slices.append(slice(lower, upper))
                lower = upper

        return ctm_sentences, parole_slices
