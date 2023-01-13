# HSC year 1 cosmic shear (reprocess)

Please cite the paper if you are using the reprocessed HSC cosmic
shear data:
https://ui.adsabs.harvard.edu/abs/2022arXiv221203257Z/abstract
Note the constrain has 0.15 sigma difference in $\Omega_m$ compared
to the official HSC year 1 analysis shown in this paper:
https://arxiv.org/pdf/1906.06041.pdf


## likelihood

Run the analysis pipeline under this directory:
```shell
cosmosis config.ini
```
### redshift distribution $n(z)$
The redshift distribution can be found in this
[notebook](notebooks/plot_chain.ipynb)

## chain

You can plot the output chain following this
[notebook](notebooks/plot_chain.ipynb)
