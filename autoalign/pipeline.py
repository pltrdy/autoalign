import importlib
import autoalign


def init_object_by_path(path, args=[], kwargs={}):
    module_name = ".".join(path.split(".")[:-1])
    class_name = path.split(".")[-1]
    try:
        module = importlib.import_module(module_name)
    except ValueError as e:
        print("Cannot import module '%s'" % module_name)
        raise e
    class_ = getattr(module, class_name)

    try:
        o = class_(*args, **kwargs)
    except TypeError as e:
        print("Could'nt instantiate class '%s' with args%s, kwargs%s"
              % (str(class_), str(args), str(kwargs)))
        raise e
    return o


def instantiate(objects):
    if isinstance(objects, str):
        objects = [objects]

    iobjects = []
    for i, object in enumerate(objects):
        if isinstance(object, str):
            iobjects += [init_object_by_path(object)]
        elif isinstance(object, tuple) or isinstance(object, list):
            path = object[0]
            args = []
            kwargs = {}

            if isinstance(object[1], dict):
                kwargs = object[1]
            else:
                args = object[1]

            if len(object) == 3:
                assert isinstance(object[2], dict)
                kwargs = object[2]

            iobjects += [init_object_by_path(path, args, kwargs)]
    return iobjects


class Pipeline:
    def __init__(self, *args):
        self.modules = [*args]

    def __call__(self, **kwargs):
        for module in self.modules:
            try:
                o = module(**kwargs)
            except BaseException:
                import sys
                debug_keys = ["ctm_paths", "docx_path"]
                msg = ("Pipeline Exception calling: '%s' with %s"
                       % (str(module), str({k: kwargs[k] for k in debug_keys})))
                raise ValueError(msg)
            kwargs.update(o)
        return o, kwargs
