"AlgebraNets" is a research paper by Jordan Hoffmann, Simon Schmitt, Simon Osindero, Karen Simonyan, Erich Elsen
https://arxiv.org/abs/2006.07360

This project was created to illustrate that a simple implementation exists for AlgebraNet convolutional layers when the objects are matrices. Indeed, they can be implemented with a single conv2d operator.

`algebranets.py` contains sample code that implements the layer according to Appendix E of the paper, and again using a single conv2d layer, and compares the results, which are the same up to floating point rounding error.

Run it like: `python3 algebranets.py`

The code uses JAX.
