[![DOI](https://zenodo.org/badge/47563497.svg)](https://zenodo.org/badge/latestdoi/47563497)

<img src="/visuals/logo.svg" width=250 align="right" />

# PyDeface
A tool to remove facial structure from MRI images.

## Dependencies:
| Package                                           | Tested version |
|---------------------------------------------------|----------------|
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) | 6.0.7.18       |
| [Python 3](https://www.python.org/downloads/)     | 3.14.3         |
| [NumPy](https://numpy.org/)                       | 2.4.2          |
| [NiBabel](https://nipy.org/nibabel/)              | 5.3.2          |
| [Nipype](https://nipype.readthedocs.io/en/latest/)| 1.11.0         |

## Installation
```
pip install pydeface
```
or
```
git clone https://github.com/poldracklab/pydeface.git
cd pydeface
pip install .
```

## How to use
```
pydeface infile.nii.gz
```

Also see the help for additional options:
```
pydeface --help
```

## License
PyDeface is licensed under [MIT license](LICENSE.txt).
