# SpatialFuser
SpatialFuster is a unified framework for fine-grained single-slice analysis and cross-sample integrative analysis across modalities including epigenomics, transcriptomics, proteomics, and metabolomics.

![Overview of SpatialFuser](https://github.com/liwz-lab/SpatialFuser/blob/main/docs/_images/SpatialFuser.png)

## Directory structure
```
.
├── Tutorials/               # Tutorial Jupyter notebooks and Python scripts
├── docs/                    # Documentation files
├── spatialFuser/            # Main Python package
├── requirements.txt         # Dependencies for installing the SpatialFuser package
├── setup.py                 # Setup script for packaging and installation
└── README.md
```


## Installation

First clone the repository. 

```
git clone https://github.com/liwz-lab/SpatialFuser.git
cd SpatialFuser
```

> **Note**
> Installing `spatialFuser` within a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is recommended.

Create a separate conda environment for installing and running SpatialFuser:

```
#create an environment called env_SpatialFuser
conda create -n env_SpatialFuser python=3.9.18

#activate your environment
conda activate env_SpatialFuser
```

> **Warning**
> The SpatialFuser package can run on both GPU and CPU. However, installation and usage on a GPU-enabled device is strongly recommended. Please note that older NVIDIA drivers may cause errors.

Install the required packages. 

For Linux

```
pip install -r requirements.txt
```

The use of the mclust algorithm requires the [rpy2 package (Python)](https://pypi.org/project/rpy2/) and the [mclust package (R)](https://cran.r-project.org/web/packages/mclust/index.html).

The [torch-geometric](https://github.com/pyg-team/pytorch_geometric#installation) library is also required.

Install SpatialFuser.

```
python setup.py build
python setup.py install
```

## [Tutorial](https://liwz-lab.github.io/SpatialFuser/)

Tutorial of SpatialFuser is [here](https://liwz-lab.github.io/SpatialFuser/).

## Support

If you have any questions, please feel free to [contact us](mailto:liweizhong@mail.sysu.edu.cn). 

## Citation

