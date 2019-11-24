#!/usr/bin/env python
"""
    Run dynamic programming on similarities object from `segment_job.py`
"""
import torch
import numpy as np


def nan_masking(tensor):
    mask = (1 - nan_mask(tensor))
    print(tensor.size())
    print(mask.size())
    return tensor[mask]


def nan_mask(tensor):
    return (tensor != tensor)


def replace_nan(tensor, value=0):
    mask = nan_mask(tensor)
    tensor[mask] = value
    return tensor


def ranged_reduce(reduce_op, tensor, range_y, range_x, empty_val=0.0, masking_fct=lambda x: x):
    tensor_slice = tensor[range_y, range_x]
    tensor_masked = masking_fct(tensor_slice)

    if sum(tensor_masked.size()) == 0:
        return torch.tensor([empty_val])

    return reduce_op(tensor_masked)


def reduce_scores(ctm_slices, docx_slices, scores, reduce_op, empty_val=0.0, masking_fct=lambda x: x):
    all_scores = []
    for i, ctm_slice in enumerate(ctm_slices):
        print("%d: " % i, str(ctm_slice))
        ctm_scores = []
        for j, docx_slice in enumerate(docx_slices):
            res = ranged_reduce(reduce_op, scores, ctm_slice,
                                docx_slice, masking_fct=masking_fct)
            score = float(res)
            print("\t%f" % score, "%d: " % j, str(docx_slice))
            ctm_scores += [score]
        all_scores += [ctm_scores]

    return torch.Tensor(all_scores)


def avg_scores(ctm_slices, docx_slices, scores):
    return reduce_scores(ctm_slices, docx_slices, scores, op=tensor_mean)


def dp(scores, score_scale_fct=lambda x: x, score_pre_scale_fct=lambda x: x, DIAGDP=False, dhw=0.9999, dgw=0.9999):
    # softmax = torch.nn.functional.softmax
    scores = score_pre_scale_fct(scores)

    # soft-normalization over CTM has some (but not much) effect
    # scores = softmax(scores, 1)

    # this soft-normalize over docx sentences which makes the alignment
    # less noise resistant i.e. docx side noise like participant list
    # will then be aligned with content.
    # scores = softmax(scores, 0)

    # haut weights (& delta)
    hw = 1
    # dhw = 1
    # dhw = 0.9999
    # # dhw = 1

    # # gauche weight (& delta)
    # # dgw = 0.9999
    # dgw = 0.9999
    # dgw = 1
    gw = 1

    # diag weight (& delta)
    diagw = 0.8

    # DIAGDP = False

    y, x = scores.size()
    t = torch.zeros([y+1, x+1])
    hist = torch.zeros([y+1, x+1])
    W = torch.ones([y+1, x+1])
    for j in range(1, x + 1):
        for i in range(1, y + 1):
            # if DIAGDP:
            #     s = None
            #     diag = t[i - 1, j - 1] + scores[i - 1, j - 1] * diagw
            #     g = t[i, j - 1] * W[i, j - 1]
            #     h = t[i - 1, j] * W[i - 1, j]

            #     if diag > g and diag > h:
            #         hist[i, j] = 2
            #         W[i, j] = 1
            #         s = diag
            #     elif g < h and hist[i - 1, j] != -1:
            #         hist[i, j] = 1
            #         if hist[i, j] == hist[i - 1, j]:
            #             W[i, j] = W[i - 1, j] * dhw
            #         else:
            #             W[i, j] = 1
            #         s = h
            #     elif g > h and hist[i, j - 1] != 1:
            #         hist[i, j] = -1
            #         if hist[i, j] == hist[i, j - 1]:
            #             W[i, j] = W[i, j - 1] * dgw
            #         else:
            #             W[i, j] = 1
            #         s = g
            #     else:
            #         hist[i, j] = 2
            #         W[i, j] = 1
            #         s = diag
            #     t[i, j] = (score_scale_fct(scores[i - 1, j - 1])) + s
            # else:
            if True:
                g = t[i, j - 1] * W[i, j - 1]
                h = t[i - 1, j] * W[i - 1, j]

                if g < h:
                    hist[i, j] = 1
                    if hist[i, j] == hist[i - 1, j]:
                        W[i, j] = W[i - 1, j] * dhw
                    else:
                        W[i, j] = 1
                    s = h
                elif g > h:
                    hist[i, j] = -1
                    if hist[i, j] == hist[i, j - 1]:
                        W[i, j] = W[i, j - 1] * dgw
                    else:
                        W[i, j] = 1
                    s = g
                else:
                    hist[i, j] = 0

                try:
                    t[i, j] = (score_scale_fct(
                        scores[i - 1, j - 1])) + max(g, h)
                except Exception as e:
                    print(score_scale_fct.__name__)
                    print(score_scale_fct(scores[i - 1, j - 1]))
                    print(max(g, h))
                    raise e

                # t[i, j] = (scores[i - 1, j - 1] ** 4) + max(g, h)

            # NOTE(09/18): rlly not sure why i used to **4
            # t[i, j] = (scores[i - 1, j - 1] ** 4) + max(g, h)
    return t, hist


def analyse_dp(hist):
    path = []
    size_y, size_x = hist.size()
    yy, xx = size_y - 1, size_x - 1
    x_count = torch.zeros(size_x)
    while yy > 1 or xx > 1:
        path += [[xx - 1, yy - 1]]
        v = hist[yy, xx]
        x_count[xx] += 1

        if v == 2 and xx > 1 and yy > 1:
            xx -= 1
            yy -= 1

        elif v < 1 and xx > 1:
            xx -= 1
            print("decay to xx: %d (yy: %d)" % (xx, yy))
        elif v > 0 and yy > 1:
            yy -= 1
            print("continue to yy: %d" % yy)
        else:
            if xx > 1:
                xx -= 1
                print("EGALITE (decay to xx: %d)" % xx)
            elif yy > 1:
                yy -= 1
                print("EGALITE (decay to yy: %d)" % yy)

    return x_count, path


def slice_len(_slice):
    stop = _slice.stop
    start = _slice.start if _slice.start is not None else 0
    step = _slice.step if _slice.step is not None else 1
    return np.ceil((stop - start) / step)


def viz_dp(cumul_scores, scores, dp_scores, dp_hist, docx_slices, ctm_slices, path, output_path="similarities.html"):
    path = [";".join([str(_) for _ in coords]) for coords in path]
    margin_left = 10
    margin_top = 10

    width = 10
    height = 10
    x_space = 2
    y_space = 2

    docx_shift = height
    ctm_shift = width

    blocks = []

    i_docx = 0
    end_docx = docx_slices[i_docx].stop
    pos_y = margin_top
    cumul_max = cumul_scores.max().item()

    for y in range(scores.size(1) - 1):

        pos_x = margin_left
        if y == end_docx:
            pos_y += docx_shift
            i_docx += 1
            if i_docx < len(docx_slices):
                end_docx = docx_slices[i_docx].stop

        i_ctm = 0
        end_ctm = ctm_slices[i_ctm].stop
        for x in range(scores.size(0) - 1):
            if x == end_ctm:
                pos_x += ctm_shift
                i_ctm += 1
                if i_ctm < len(ctm_slices):
                    end_ctm = ctm_slices[i_ctm].stop
            score = scores[x, y]
            coord = "%d;%d" % (y, x)

            _class = ""
            if coord in path:
                _class = " class='path' "

            if score != 0.0 or coord in path:
                # print(score)
                gray = 255 * (1 - score)
                olor = "background-color: rgb(%d, %d, %d);" % (gray,
                                                               gray, gray)
                position = "top: %dpx;" % pos_y
                position += "left: %dpx;" % pos_x

                block = "<div %s style='%s'></div>" % (
                    _class, " ".join([color, position]))
                blocks += [block]

            pos_x += width + x_space
        blocks += ["\n"]
        pos_y += height + y_space

    style = """<style>
                    div {
                        position: absolute;
                        width: %dpx;
                        height: %dpx;
                    }

                    .path {
                        -webkit-box-sizing: border-box;
                        box-sizing: border-box;
                        -moz-box-sizing: border-box;
                        border: 2px solid green;
                    }
                </style>
            """ % (width, height)

    head = "<head>\n" + style + "\n</head>\n"
    body = "<body>\n" + "".join(blocks) + "</body>"
    html = "<html>" + head + body + "</html>"
    with open(output_path, 'w') as f:
        print(html, file=f)
    print("Viz_dp output: %s" % output_path)


def align_slices(dp_path, docx_slices, ctm_slices, one_doc_per_ctm=False, verbose=False):
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def inslice(val, s):
        start = 0 if s.start is None else s.start
        step = 1
        stop = s.stop
        return (val >= start and val < stop)

    print("Alignment")

    i_docx_slice = len(docx_slices) - 1
    i_ctm_slice = len(ctm_slices) - 1
    i_coord = 0

    # feature meant not to replicate CTM over 2 docx slice by
    # giving it to the best.
    prev_count = 0
    cur_count = 0
    check_count = False

    cur_slices = []
    processed_slices = []
    aligned_slices = [cur_slices]
    log("First slices:")
    log("* docx[#%d]: " % i_docx_slice, docx_slices[i_docx_slice])
    log("* ctm[#%d]: " % i_ctm_slice, ctm_slices[i_ctm_slice])

    def find_prev_aligned():
        for i in range(len(aligned_slices)-2, 1, -1):
            if len(aligned_slices[i]) > 0:
                return aligned_slices[i][-1]

    while i_coord < len(dp_path):
        y, x = dp_path[i_coord]
        log("x, y: ", x, y)

        if i_docx_slice > 0 and not inslice(y, docx_slices[i_docx_slice]):
            # new doc slice on position
            log("[NEW DOCX SLICE] %d is over w/ %s" %
                (i_docx_slice, str(cur_slices)))

            if one_doc_per_ctm and len(cur_slices) == 1 and len(aligned_slices) >= 2:
                # only 1 CTM means no CTM border, therefore we must check_count a bit
                log("CHECK-COUNT: new docx slice after a len=1 one")
                i_prev_aligned_doc = max(len(aligned_slices) - 2, 0)
                while i_prev_aligned_doc > 0:
                    if len(aligned_slices[i_prev_aligned_doc]) >= 1:
                        break
                    i_prev_aligned_doc -= 1

                if cur_count > prev_count:
                    log("\t!! [%d|%d] higher cur_count" %
                        (cur_count, prev_count))
                    log("\t removing from align[%d|%d](%s), keeping in cur" % (
                        i_prev_aligned_doc, i_prev_aligned_doc-len(aligned_slices), str(aligned_slices[-2])))
                    _ = aligned_slices[i_prev_aligned_doc].pop()
                    if not _ in cur_slices:
                        cur_slices.insert(0, _)

                    log("\tswapping prev_count <- cur_count")
                    prev_count = cur_count
                else:
                    log("\t!! [%d|%d] lower cur_count" %
                        (cur_count, prev_count))
                    log("\t removing from cur_slices %s" % str(cur_slices))
                    cur_slices.pop()
                    log("\t keeping prev_count=%d" % prev_count)
            log("\t (resetting cur_count")
            cur_count = 0

            log("\t (resetting/appending cur_slices)")
            fpa = find_prev_aligned()
            if fpa is not None:
                try:
                    if len(cur_slices) > 0:
                        if one_doc_per_ctm:
                            assert cur_slices[0] + \
                                1 == fpa, "%d != %d" % (cur_slices[0]+1, fpa)
                        else:
                            assert fpa in [cur_slices[0], cur_slices[0]+1]
                except AssertionError as e:
                    log("aligned_slices[-1]: '%s'" % str(aligned_slices[-1]))
                    log("aligned_slices[-2]: '%s'" % str(aligned_slices[-2]))
                    log("aligned_slices[-3]: '%s'" % str(aligned_slices[-3]))
                    raise e
            cur_slices = []
            aligned_slices.append(cur_slices)
            i_docx_slice -= 1
            log("\t (#aligned_slices: %d)" % len(aligned_slices))

        elif i_ctm_slice >= 0 and (not inslice(x, ctm_slices[i_ctm_slice]) or i_coord+1 == len(dp_path)):
            # new ctm slice on position
            log("[NEW CTM SLICE] (#%d is over)" % (i_ctm_slice))

            if one_doc_per_ctm:
                if len(cur_slices) == 1:
                    # Only check for 1rst one (not mandatory condition)
                    log("[CHECK-COUNT: first CTM")
                    # find relevant previous doc index
                    i_prev_aligned_doc = max(len(aligned_slices) - 2, 0)
                    while i_prev_aligned_doc > 0:
                        if len(aligned_slices[i_prev_aligned_doc]) >= 1:
                            break
                        i_prev_aligned_doc -= 1
                    log("\t(using i_prev_aligned_doc=%d (i.e. %d)" % (
                        i_prev_aligned_doc, i_prev_aligned_doc-len(aligned_slices)))
                    if len(aligned_slices[i_prev_aligned_doc]) >= 1:
                        log("\t(this prev_align has >= 1 ctm, continuing)")
                        if cur_slices[-1] == aligned_slices[i_prev_aligned_doc][-1]:
                            log(
                                "\t(last i_ctm match for cur/prev_align, continuing)")
                            if cur_count > prev_count:
                                log("\t!! [%d|%d] higher cur_count" %
                                    (cur_count, prev_count))
                                log("\t removing from prev_align %s, keeping in cur" % str(
                                    aligned_slices[i_prev_aligned_doc]))
                                _ = aligned_slices[i_prev_aligned_doc].pop()
                                if not _ in cur_slices:
                                    cur_slices.insert(0, _)
                                # log("\t also reseting prev_count")
                                # prev_count = 0
                            else:
                                log("\t!! [%d|%d] lower cur_count" %
                                    (cur_count, prev_count))
                                log("\t removing from cur, keeping in prev_align")
                                _ = cur_slices.pop()
                                assert _ in aligned_slices[i_prev_aligned_doc]
                                # log("\t keeping prev_count=%d" % prev_count)
                        else:
                            log("! aborting check_count: i_ctm does not match")
                    else:
                        log("! aborting check_count: prev_align has no ctm")
            log("\t resetting both prev/cur count")
            prev_count = 0
            cur_count = 0
            i_ctm_slice -= 1
            log("\t i_ctm_slice=%d" % i_ctm_slice)
        else:
            # continue on path
            i_coord += 1
            log("[CONTINUE] on path w/ i_coord:%d [%d]" %
                (i_coord, len(dp_path)))

        if i_ctm_slice not in cur_slices and i_ctm_slice >= 0:
            log("\t(add i_ctm_slice: %d un cur_slices)" % i_ctm_slice)
            cur_slices.append(i_ctm_slice)

        cur_count += 1

    for last_i_docx_slice in range(i_docx_slice, 0, -1):
        # add list docx slices that have been ignored?
        aligned_slices.append([])

    log("last slices: %s" % str(cur_slices))
    aligned_slices = [slices[::-1] for slices in aligned_slices[::-1]]

    log(str(one_doc_per_ctm), aligned_slices)
    log(len(aligned_slices), len(docx_slices))
    log(len(sum(aligned_slices, [])), len(ctm_slices))
    log(len(set(sum(aligned_slices, []))))
    if one_doc_per_ctm:
        assert_msgs = []
        assert_msgs += ["#aligned_slices: %d" % len(aligned_slices)]
        assert_msgs += ["#ctm_slices: %d" % len(ctm_slices)]
        assert_msgs += ["#docx_slices: %d" % len(docx_slices)]
        assert_msgs += ["#sum(aligned_slices): %d" %
                        len(sum(aligned_slices, []))]
        assert_msgs += ["#set(sum(aligned_slices)): %d" %
                        len(set(sum(aligned_slices, [])))]
        assert_msgs += ["i_docx_slice: %d, i_ctm_slice: %d" %
                        (i_docx_slice, i_ctm_slice)]
        assert_msgs += [str(docx_slices)]
        assert_msgs += [str(aligned_slices)]
        assert len(aligned_slices) == len(docx_slices) and len(
            sum(aligned_slices, [])) == len(ctm_slices), "\n".join(assert_msgs)

    return aligned_slices


def viz_alignement(scores, slices_align, docx_slices, ctm_slices, docx_sentences,
                   ctm_sentences, output_html="alignment.html", output_pt="alignment.pt",
                   extra_data={}):
    """
    Args:
        slices_align(list[list[int]]): alignment of docx to ctm.
            slices_align[i] contains id j such that
            ctm_slices[j] is aligned with docx_slices[i]
            extra_data: dict to save in output_pt in the first record 
                        i.e. for all (k, v) in extra_data there will be
                        align[0][k] = v, align being saved to `output_pt`
    """
    import html

    def sentences2scores(sentences):
        return [s for sentence in sentences for w, s in sentence]

    def sentences2text(sentences, with_scores=False):
        if with_scores:
            text = " ".join([" ".join([w for w, s in sentence])
                             for sentence in sentences])
        else:
            text = " ".join([" ".join(sentence)
                             for sentence in sentences])
        text = " ".join(text.split())
        return text

    blocks = []
    align = []
    for i_docx_slice, i_ctm_slices in enumerate(slices_align):
        _docx_text = html.escape(sentences2text(
            docx_sentences[docx_slices[i_docx_slice]]))

        _ctm_slices_sentences = [ctm_sentences[ctm_slices[i_ctm_slice]]
                                 for i_ctm_slice in i_ctm_slices]
        _ctm_text = "<br/><br/>".join(
            [html.escape(sentences2text(_ctm_sentences, with_scores=True))
             for _ctm_sentences in _ctm_slices_sentences])

        _docx_block = '\n<p class="doc">\n' + _docx_text + '\n</p>'
        _ctm_block = '\n<p class="ctm">\n' + _ctm_text + '\n</p>'
        block = '<div class="block">' + _ctm_block + _docx_block + '</div>'
        blocks += [block]
        align += [{"ctm": [sentences2text(_ctm_sentences, with_scores=True)
                           for _ctm_sentences in _ctm_slices_sentences],
                   "doc": sentences2text(docx_sentences[docx_slices[i_docx_slice]]),
                   "ctm_scores": [sentences2scores(_ctm_sentences)
                                  for _ctm_sentences in _ctm_slices_sentences]
                   }]

    script = ""
    css = {
        ".block": ["width: 100%", "text-align: justify;", "background-color: white",
                   "display: inline-block", "margin-top: 10px"],
        "p": ["width: 49%"],
        "p.ctm": ["float: right"]
    }
    css_code = "\n".join(
        ["%s{\n\t%s\n}" % (k, "\n\t".join(["%s;" % instr
                                           for instr in v]))
         for k, v in css.items()])
    style = "<style>\n" + css_code + "\n</style>"

    meta = '\n<meta charset="utf-8" />'
    title = "\n<title>Aligned segments</title>"
    head = "<head>" + title + meta + script + style + "\n</head>"
    body = "\n<body>\n" + "\n<hr>".join(blocks) + "\n</body>"
    page = "<html>\n" + head + body + "</html>"

    if output_html is not None:
        with open(output_html, 'w') as out:
            out.write(page)
        print("Wrote html to: '%s'" % output_html)

    if output_pt is not None:
        for k, v in extra_data.items():
            align[0][k] = v
        with open(output_pt, 'wb') as out:
            torch.save(align, out)
        print("Wrote alignment to: '%s'" % output_pt)
