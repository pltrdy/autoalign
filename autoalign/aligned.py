import json
import os


def get_public_meetings():
    try:
        import public_meetings
        return public_meetings.data.data_root

    except ImportError:
        raise ImportError("`public_meetings` package is missing, "
            "you can install it with `pip3 install public_meetings`")


ALIGNED = {
    "public_meetings": get_public_meetings
}


def get_aligned_name(aligned_name):
    aligned_name = aligned_name.lower().strip()

    get_aligned = ALIGNED.get(aligned_name)
    if get_aligned is None:
        raise ValueError("Unknow aligned '%s', choices are: %s"
                         % (aligned_name, str(ALIGNED.keys())))
    else:
        return get_aligned()


def get_aligned_dir(aligned_dir):
    assert os.path.isdir("aligned_dir")
    return aligned_dir


def get_aligned_args(args):
    """Loading aligned from args namespace from argparse
    """
    if args.aligned_dir is not None:
        if args.aligned_name is not None:
            raise ValueError(
                "args -aligned_dir and -aligned_name should not be both set")
        return get_aligned_dir(args.aligned_dir)
    elif args.aligned_name is not None:
        return get_aligned_name(args.aligned_name)
    else:
        raise ValueError(
            "No aligned argument. Set -aligned_dir xor -aligned_name")
