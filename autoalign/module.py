import pickle


class Module(object):
    def __init__(self):
        self.return_keys = set()
        pass

    def assert_return_keys(self, o):
        assert all([k in o.keys() for k in self.return_keys])

    def __call__(self, *args, save=None, **kwargs):
        o = self.process(*args, **kwargs)
        self.assert_return_keys(o)

        if save is not None:
            with open(save, 'wb') as f:
                pickle.save(o, f)

        return o

    def process(self, *args, **kwargs):
        return self.do_process(*args, **kwargs)

    def do_process(self, *args, **kwargs):
        raise NotImplementedError()
