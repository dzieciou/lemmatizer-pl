SCWAD-tagger: Toygger
=====================

The fork of `Toygger tagger`_, described in:

* Krasnowska-Kieraś Katarzyna (2017). `Morphosyntactic disambiguation for Polish with bi-LSTM neural networks`_.
  In Zygmunt Vetulani and Patrick Paroubek (eds.) *Proceedings of the 8th Language &
  Technology Conference: Human Language Technologies as a Challenge for Computer Science and
  Linguistics*, pp. 367–371, Poznań, Poland. Fundacja Uniwersytetu im. Adama Mickiewicza w Poznaniu.

.. _Toygger tagger: http://mozart.ipipan.waw.pl/~kkrasnowska/PolEval/src/SCWAD-tagger/
.. _Morphosyntactic disambiguation for Polish with bi-LSTM neural networks: http://ltc.amu.edu.pl/book/papers/PolEval1-2.pdf

Setup
-----

Windows/Conda
~~~~~~~~~~~~~

We're assuming you have:

* Windows 10
* `Microsoft Visual C++ Build Tools`_
* Conda 64bit package manager
* Latest `OpenSSL`_
* `Morfeusz2`_ 64bit version

.. _Microsoft Visual C++ Build Tools: https://visualstudio.microsoft.com/pl/downloads/
.. _OpenSSL: https://slproweb.com/products/Win32OpenSSL.html
.. _Morfeusz2: http://sgjp.pl/morfeusz/dopobrania.html

To install further requirements:

.. code::console

    conda create -n venv python=3.6
    conda activate venv
    pip install -r requirements.txt
    conda install -c anaconda pydot

    python -m easy_install http://sgjp.pl/morfeusz/download/20181014/ubuntu-xenial/morfeusz2-0.4.0-py3.6-win-amd64.egg

Training
--------

Training data format
~~~~~~~~~~~~~~~~~~~~

Tab-separated file with the following columns:

* wordform
* possible morphological analyses, pipe-separated (e.g. ``subst:sg:inst:n|ger:sg:inst:n:perf:aff``)
* correct morphological analysis
* token label (may be left empty except for ``IGN`` for unknown words)
* (unused)
* (unused)

Use ``PolEval/prepare_data.py`` to convert XCES to this format.
Use ``dag2training.py`` to convert Morfeusz/Concraft DAG to this format.

XCES training corpora can be downloaded from `PolEval 2017`_.

.. _PolEval 2017: http://2017.poleval.pl/index.php/tasks/


Required resources
~~~~~~~~~~~~~~~~~~

* If Concraft2 input is to be included in input features:
    - Concraft2 binary (set ``CONCRAFT_BIN`` in ``concraft_utils.py`` accordingly)
    - tagset config file
    - a directory where created Concraft data will be placed (``DATA_DIR`` in ``concraft_utils.py``)
* If Word2Vec embeddings are to be included in input features:
    - `pretrained wordembeddings`_ created by Aleksander Wawer, available from  (set ``WORD2VEC_MODELS_DIR``
       in ``data.py`` to the directory where the wordembeddings reside, do not change the model files’ names)

.. _pretrained wordembeddings: http://mozart.ipipan.waw.pl/~axw/wordembeddings/orth/

Running training
~~~~~~~~~~~~~~~~

Use ``training.py`` for tagging. To get help information:

.. code::console

    training.py -h

More options available by modifying ``settings.py``.

+------------------------------------------------------------------------------+
| The Keras library used by the script can be greedy on CPU cores.             |
| Wrap the call e.g. in taskset to restrict the number of cores used.          |
+------------------------------------------------------------------------------+

Define your tagset by modifying tagset.py.


Tagging
-------

Use ``tagging.py`` for tagging. To get help information:

.. code::console

    tagging.py -h

Currently supported input formats:

* XCES
