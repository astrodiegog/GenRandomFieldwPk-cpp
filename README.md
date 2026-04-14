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




