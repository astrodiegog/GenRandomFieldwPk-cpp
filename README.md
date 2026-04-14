# GenRandomFieldwPk-cpp
Generate a Gaussian random field and apply power spectrum in C++

Given a parameter file that specifies a logarithmic power spectrum $P(k) = A_s (k/k_s)^{n_s}$ (where $A_s$ is given in units of $(h/Mpc)^{n_dim}$), we generate a zero-mean Gaussian random field and convolve it with the power spectrum, following [Bertschinger, E (2001)](https://ui.adsabs.harvard.edu/abs/2001ApJS..137....1B/abstract).

The parameter file expected for this calculation requires the following fields

```
ndims=
Lbox=
Ng=
As=
ks=
ns=
```

where `n_dims` specifies the number of dimensions (currently max of three), `Lbox` describes the length (in units of $Mpc/h$) along one dimension, `Ng` specifies the number of cells along a dimension, and the form of the power spectrum is fully described by `As`, `ks`, and `ns`.


To generate the Gaussian random field, we use the `mt19937` PRNG that comes with C++ [random](https://en.cppreference.com/w/cpp/numeric/random.html). We have the process with `procID == 0` serving as the random number generator, and the other processes requesting random numbers. The random number generator process fills itself up first. While we could generate a seperate random number engine for each process by giving each a different seed (say `local_seed = global_seed * (procID + 1)`), we don't know whether these different seeds would be correlated with one another. They generally won't be, but better to stick with just one generator.

In the future, the random number generation can be improved by using counter-based PRNG (see pedagogical paper [here](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf). Specifically, we could use the [`std::philox_engine`](https://en.cppreference.com/w/cpp/numeric/random/philox_engine.html). While it was first proposed in C+11, it has been only been part of the standard since C++26. While there are other implementations (see [here](https://github.com/johnsalmon/cpp-counter-based-engine) ), this current repo is mostly about applying and then re-calculating a power spectrum on top of a Gaussian random field. In effect, I want to keep it light-weight without extra libraries, at the expense of performance during creation of the random field.

We could also provide a global seed into the params file



