import logging
import random as rn
from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm import tqdm

log = logging.getLogger(__name__)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def make_reproducible():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def timing(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = orig_func(*args, **kwargs)
        elapsed = time.time() - start
        caller = orig_func.__qualname__
        log.debug(f'{caller} ran in: {elapsed:.2f} sec')
        return result

    return wrapper
