# COPNN: Copula-based Neural Networks

This is the code for our AISTATS '25 paper, "Flexible Copula-Based Mixed Models in Deep Learning: A Scalable Approach to Arbitrary Marginals".

For full implementation details see the paper.

For running the simulations use the `simulate.py` file, like so:

```
python simulate.py --conf conf_files/conf_categorical_continuous.yaml --out res.csv
```

The `--conf` attribute accepts a yaml file such as `conf_categorical_continuous.yaml` which you can configure.

To run various real data experiments see the jupyter notebooks in the notebooks folder. We cannot unfortunately attach the actual datasets, see paper for details.

For using COPNN with your own data use the `COPNLL` loss layer as shown in notebooks and simulation.
