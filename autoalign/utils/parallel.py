"""
    https://gist.githubusercontent.com/oseiskar/dbd38098038df8944e21b41c42668440/raw/3edefc436bab14b506ada207726a4a46aa8cccd9/pathos-multiprocessing-example.py
"""
from pathos.multiprocessing import ProcessingPool as Pool


def parallel_map(func, array, n_workers):
    def compute_batch(i):
        try:
            return func(i)
        except KeyboardInterrupt:
            raise RuntimeError("Keyboard interrupt")

    p = Pool(n_workers)
    err = None
    # pylint: disable=W0703,E0702
    # some bs boilerplate from StackOverflow
    try:
        return p.map(compute_batch, array)
    except KeyboardInterrupt as e:
        print('got ^C while pool mapping, terminating the pool')
        p.terminate()
        err = e
    except Exception as e:
        print('got exception: %r:, terminating the pool' % (e,))
        p.terminate()
        err = e

    if err is not None:
        raise err
