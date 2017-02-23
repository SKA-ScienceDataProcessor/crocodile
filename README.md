
Crocodile -- Interferometry Imaging Algorithm Reference Library
===============================================================

This is a project to create a reference code in NumPy for somewhat
simplified aperture synthesis imaging. Check the
[AW-gridding kernel work description](KERNEL_WORK.md) for information
about kernel prototyping efforts based on this repository.

Warning: The current code is an experimental proof-of-concept. More
here soon.

Motivation
----------

The Crocodile algorithm reference library is designed to present
imaging algorithms in a simple Python-based form. This is so that the
implemented functions can be seen and understood without resorting to
interpreting source code shaped by real-world concerns such as
optimisations.

Requirements
------------

This library is built using Python 3.0. We use the following libraries:

  * `jupyter` - for example notebooks
  * `numpy` - for calculations
  * `matplotlib` - for visualisation
  * `pyfits` - for reading reference data

You will have to install these dependencies, either manually using
your package manager of choice or using `pip`:

     $ pip install -r requirements.txt

Acquiring data
--------------

We are using GitHub's
[large file storage](https://git-lfs.github.com/) (LFS) to store data
files. To pupluate the `data/` directory you will need to have it
installed and activated for this repository. So for example:

```bash
    $ git lfs install
    $ git lfs pull
```

After `git-lfs` has finished downloading, the required files should
now appear in the `data/` directory.

Orientation
-----------

The content of this project is meant for learning and experimentation,
not usage. If you are here to learn about the process of imaging, here
is a quick guide to the project:

  * `crocodile`: The main Python source code
  * `examples`: Usage examples, mainly using Jupyter notebooks.
  * `docs`: Complete documentation. Includes non-interactive output of examples.
  * `data`: Data used

Running Notebooks
-----------------

Jupyter notebooks end with `.ipynb` and can be run as follows from the
command line:

     $ jupyter notebook examples/notebooks/wkernel.ipynb

Building documentation
----------------------

For building the documentation you will need Sphinx as well as
Pandoc. This will extract docstrings from the crocodile source code,
evaluate all notebooks and compose the result to form the
documentation package.

You can build it as follows:

    $ make -C docs [format]

Omit [format] to view a list of documentation formats that Sphinx can
generate for you.
