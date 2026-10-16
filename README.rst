==========================================
TensorPyPy / MetaTensor (public snapshot)
==========================================

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.23298877.svg
  :target: https://doi.org/10.5281/zenodo.23298877

+----------+---------+
| preprint | |doi|   |
+----------+---------+

TensorPyPy is a fork of PyPy. Its RPython meta-tracing JIT carries a tensor
runtime, MetaTensor. The code is in ``rpython/metatensor/``,
``pypy/module/_metatensor/`` and ``lib_pypy/tensorpypy/``.

This repository is a snapshot of the development repository. It has no
history, no benchmark results, no raw data and no internal notes.

Benchmarks and experiments are described in ``benchmark/README.md``.

License
-------

Upstream PyPy files are under ``LICENSE``. The TensorPyPy and MetaTensor
additions, including everything under ``benchmark/``, are under the MIT
license in ``LICENSE-tensorpypy``, copyright Yusuke Izawa.

The PyPy README follows.

=====================================
PyPy: Python in Python Implementation
=====================================

Welcome to PyPy!

PyPy is an interpreter that implements the Python programming language, based
on the RPython compiler framework for dynamic language implementations.

The home page for the interpreter is:

    https://pypy.org/

If you want to help developing PyPy, this documentation might help you:

    https://doc.pypy.org/

More documentation about the RPython framework can be found here:

    https://rpython.readthedocs.io/

The source for the documentation is in the pypy/doc directory.


Using PyPy instead of CPython
-----------------------------

Please read the information at https://pypy.org/ to find the correct way to
download and use PyPy as an alternative to CPython.


Building
--------

Building PyPy is not the recommended way to obtain the PyPy alternative python
interpreter. It is time-consuming and requires significant computing resources.
More information can be found here:

    https://doc.pypy.org/build.html

Enjoy and send us feedback!

    the pypy-dev team <pypy-dev@python.org>
