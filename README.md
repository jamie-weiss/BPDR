# BPDR
Dimensionality Reduction Algorithm for Exploratory Analyses

### Instalation
1. Clone the repository from github
2. Navigate to the repo's directory and install required packages using `pip install -r requirements.txt`

### Usage
In a new file where BPDR is going to be used, the following command will sucessfully import the package:
`from <path/to/bitpack.py>/bitpack.py import BPDR`

Initialize a new instance of BPDR with a desired n_components:

`bpdr = BPDR(n_components=2)`

Fit and transform the bpdr object on the data to be reduced:

`bpdr_data = bpdr.fit_transform(iris_data, iris_targets)`

Print out the variance attribute to check overall explanation:

`bpdr.variances`
