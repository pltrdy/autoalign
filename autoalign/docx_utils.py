import docx


def is_p_elmt(e):
    return type(e) == docx.oxml.CT_P


def is_tbl_elmt(e):
    return type(e) == docx.oxml.CT_Tbl


def is_row_elmt(e):
    return type(e) == docx.oxml.CT_Row


def docx_iter(root):
    if type(root) == docx.document.Document:
        root = root.element

    elmts = []
    for children in root.iterchildren():
        if is_p_elmt(children):
            yield children
        elif is_tbl_elmt(children):
            yield children
        else:
            yield from docx_iter(children)


def elmt2txt(elmt):
    if is_p_elmt(elmt):
        return p2txt(elmt)
    elif is_tbl_elmt(elmt):
        return tbl2txt(elmt)
    else:
        raise ValueError("Nothing to do for type '%s'" % str(type(elmt)))


def p2txt(p_elmt, debug=False):
    def log(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    txts = []
    prev_txt = ""
    for children in p_elmt.iterchildren():
        if type(children) == docx.oxml.text.run.CT_R:
            # we only take first text of CT_R (repetitions otherwise)
            txt = next(children.itertext())
            log("CT_R: text [%s]" % txt)
        else:
            txt = "".join(list(children.itertext()))

        if "".join(txt.split()) != "".join(prev_txt.split()):
            log("text accepted: %s" % txt)
            txts += [txt]
        else:
            log("text rejected! %s" % txt)
            pass
            # log("cancelling " + txt)
        prev_txt = txt
    return "".join([_ for _ in txts])


def _p2txt(p_elmt):
    txts = []
    prev_txt = ""
    for children in p_elmt.iterchildren():
        if type(children) == docx.oxml.text.run.CT_R:
            # we only take first text of CT_R (repetitions otherwise)
            txt = " ".join(next(children.itertext()).split())
        else:
            txt = "".join(list(children.itertext()))

        if "".join(txt.split()) != "".join(prev_txt.split()):
            txts += [txt]
        else:
            pass
            # print("cancelling " + txt)
        prev_txt = txt
    return " ".join([_.strip() for _ in txts])


def tbl2txt(tbl_elmt):
    txts = []
    for children in tbl_elmt.iterchildren():
        # print("# New CH")
        if is_row_elmt(children):
            prevcelltxt = ""
            for cell in children.iterchildren():
                # print("## New Cell")
                cell_txt = ""
                prevtxt = ""
                for txt in cell.itertext():
                    # print("#### ", txt, end="")
                    if txt != prevtxt:
                        # print("valid")
                        cell_txt += " %s " % txt
                    else:
                        # print("canceled")
                        pass
                    prevtxt = txt

                cell_txt = " ".join(cell_txt.split())
                if cell_txt != " ".join(prevcelltxt.split()):
                    txts += [cell_txt]
                prevcelltxt = cell_txt
    return " ".join(txts)
