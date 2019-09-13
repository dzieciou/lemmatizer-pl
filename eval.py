import logging
import random as rn
from contextlib import contextmanager
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


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


@contextmanager
def timing(description: str) -> None:
    """
    Measures and logs duration of the task,
    :param description: description to log before and after completing the task
    :return:
    """
    log = logging.getLogger()
    log.debug(f'{description}...')
    start = time()
    yield
    elapsed_time = time() - start
    log.debug(f'{description} COMPLETED in {elapsed_time:.2f}s')
