import json


def get_public_meetings():
    try:
        import public_meetings
        return public_meetings.data.make_mapping()

    except ImportError:
        raise ImportError("`public_meetings` package is missing, "
            "you can install it with `pip3 install public_meetings`")


MAPPINGS = {
    "public_meetings": get_public_meetings
}


def load_mapping_name(mapping_name):
    mapping_name = mapping_name.lower().strip()

    make_mapping = MAPPINGS.get(mapping_name)
    if make_mapping is None:
        raise ValueError("Unknow mapping '%s', choices are: %s"
                         % (mapping_name, str(MAPPINGS.keys())))
    else:
        return make_mapping()


def load_mapping_path(mapping_path):
    with open(mapping_path) as f:
        mapping = json.load(f)

    assert isinstance(mapping, dict)

    for h, meeting in mapping:
        assert "doc" in meeting.keys(), "No doc for meeting '%s'" % h
        assert "ctm" in meeting.keys(), "No ctm for meeting '%s'" % h
        assert len(meeting["ctm"]) > 0, "No ctm for meeting '%s'" % h

    return mapping


def load_mapping_args(args):
    return load_mapping_both(
        mapping_path=args.mapping_path,
        mapping_name=args.mapping_name
    )


def load_mapping_both(mapping_path=None, mapping_name=None):
    if mapping_path is not None:
        if mapping_name is not None:
            raise ValueError(
                "mapping_path and mapping_name should not be both set")
        return load_mapping_path(mapping_path)
    elif mapping_name is not None:
        return load_mapping_name(mapping_name)
    else:
        raise ValueError(
            "No mapping argument. Set mapping_path xor mapping_name")
