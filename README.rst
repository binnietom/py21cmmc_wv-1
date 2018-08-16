========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |coveralls|
        | |codacy|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/py21cmmc_wv/badge/?style=flat
    :target: https://readthedocs.org/projects/py21cmmc_wv
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/steven-murray/py21cmmc_wv.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/steven-murray/py21cmmc_wv

.. |coveralls| image:: https://coveralls.io/repos/steven-murray/py21cmmc_wv/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/steven-murray/py21cmmc_wv

.. |codacy| image:: https://img.shields.io/codacy/REPLACE_WITH_PROJECT_ID.svg
    :target: https://www.codacy.com/app/steven-murray/py21cmmc_wv
    :alt: Codacy Code Quality Status

.. |version| image:: https://img.shields.io/pypi/v/py21cmmc_wv.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/py21cmmc_wv

.. |commits-since| image:: https://img.shields.io/github/commits-since/steven-murray/py21cmmc_wv/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/steven-murray/py21cmmc_wv/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/py21cmmc_wv.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/py21cmmc_wv

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/py21cmmc_wv.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/py21cmmc_wv

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/py21cmmc_wv.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/py21cmmc_wv


.. end-badges

A set of python bindings for 21cmFAST allowing native Python plugins for 21cmMC.

Note: this package has just started. It doesn't do any MCMC at the moment.

* Free software: MIT license

Installation
============

Just do (not actually online yet!)::

    pip install py21cmmc_wv

Or to get the bleeding edge::

    pip install git+git://github.com/steven-murray/py21cmmc_wv.git

For development, it is easiest to do (from top-level directory of this package)::

    pip install -e .

Quick Usage
===========

This is meant as a plugin for ``py21cmmc``. You should read the docs for that package first. The primary functionality
added by this plugin is a set of ``Likelihood``s which can be passed directly to the ``run_mcmc`` function of
``py21cmmc`` as the likelihoods controlling the MCMC.

The likelihoods contained in this plugin are dependent on having the core modules from ``py21cmmc_fg`` loaded in the
``CosmoHammer`` ``ChainContext``.

In addition to providing simple plugin capability, the likelihoods here also contain a simulation method which can be
used to *create* mock data identical to that which it claims to fit.

Documentation
=============

To view the docs, install the ``requirements_dev.txt`` packages, go to the docs/ folder, and type "make html", then
open the ``index.html`` file in the ``_build/html`` directory.


To run the all tests run (no tests as yet...)::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
