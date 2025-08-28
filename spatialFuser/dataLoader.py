import numpy as np
import scanpy as sc
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader, ClusterData, ClusterLoader
import os
from .utils import get_spatial_adj


class H5adLoader:
    """
    read and preprocess .h5ad data

    Args:
        data_dir: str. Directory containing the `.h5ad` dataset.

    Functions:
        __getitem__: Load the `.h5ad` file.
        log_and_norm: Perform normalization and log-transformation.

    Returns:
        The processed AnnData object.
    """

    def __init__(self, data_dir):
        """
        :param data_dir: (string) directory containing the h5ad dataset
        """
        self.data_dir = data_dir
        self.all_filenames = os.listdir(data_dir)

    def __len__(self):
        # return size of dataset
        return self.filenames.shape[0]

    def __getitem__(self, files):
        """
        :param files: the .h5ad files to be load
        :return: a list of anndata
        """
        adata_list = []
        self.filenames = np.array([os.path.join(self.data_dir, f) for f in files if f.endswith('.h5ad')])
        for file in self.filenames:
            assert 0 < self.__len__(), 'number of files to be loaded should more than 0'
            adata_list.append(sc.read_h5ad(file))
        return adata_list

    @staticmethod
    def log_and_norm(adata, data_tech, n_svgs=2000):
        if data_tech == 'seq-based':
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_svgs)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.raw = adata
            adata = adata[:, adata.var['highly_variable']]
            return adata
        elif data_tech == 'image-based':
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            return adata
        else:
            raise ValueError('Please choose the technique type of your deta: seq-based or image-based')


class SpatialFuserDataLoader:
    """
    Extract information from AnnData to create a PyG object.

    Args:
        args: argparse.Namespace. Global hyperparameters.
        data_dir: str. Directory containing the `.h5ad` dataset.
        data_tech: str. Technology type of the dataset, either "seq-based" or "image-based".
        files: list of str. The `.h5ad` files to be loaded.

    Functions:
        load_adata: Reads `.h5ad` files via `h5ad_loader.__getitem__`.
        rename_label: Adds a specified renamed label column to `anndata.obs`.
        remove_type: Removes a specified category from a given label column.
        remove_nan: Deletes spots with missing values in a specified label column.
        pre_processing: Performs feature extraction, normalization, spatial coordinate normalization, adjacency graph construction, and batch annotation.
        create_pyg_data: Converts the preprocessed AnnData object into a PyG data object.
    """

    def __init__(self,
                 args,
                 data_dir='/public8/lilab/student/whcai/Integration/data/10X_DLPFC',
                 data_tech='seq-based',
                 files: list = None):
        """
        :param args: Globe hyperparameters
        :param data_dir: (string) directory containing the h5ad dataset
        :param data_tech: technology type of the dataset (seq-based/image-based)
        :param files: the .h5ad files to be load
        """
        self.args = args
        self.files = files
        self.data_tech = data_tech
        self.h5ad_loader = H5adLoader(data_dir)
        self.loader = None
        self.adata_list = []
        self.adata = None
        self.data = None

    def load_adata(self):
        self.adata_list = self.h5ad_loader.__getitem__(self.files)

    def rename_label(self, original_label='cell_type', target_label='Region'):
        for adata in self.adata_list:
            adata.obs[target_label] = adata.obs[original_label]

    def remove_type(self, obs_name='Region', target='Excluded'):
        self.adata_list = [adata[adata.obs[obs_name] != target] for adata in self.adata_list]

    def remove_nan(self, obs_name='Region'):
        self.adata_list = [adata[~adata.obs[obs_name].isna()] for adata in self.adata_list]

    def pre_processing(self, n_svgs=2000, k_cutoff=10, batch_label: list = None):
        if batch_label is None:
            batch_label = [i for i in range(len(self.files))]
        assert (len(self.files) == len(batch_label)), 'Number of labels should be equal to number of datasets'
        assert len(self.adata_list) > 0, 'Please run SpatialFuserDataLoader.load_adata first.'

        spatial_adj_list = []
        for i in range(len(self.adata_list)):
            temp_adata = self.adata_list[i]
            # log and norm
            temp_adata = self.h5ad_loader.log_and_norm(temp_adata, data_tech=self.data_tech, n_svgs=n_svgs)
            # Unique barcode
            temp_adata.obs_names = temp_adata.obs_names + '_' + 'batch' + '_' + str(1)
            # make batch label
            temp_adata.obs['batches'] = batch_label[i]
            # get Nat Info
            temp_adata = get_spatial_adj(temp_adata, k_cutoff=k_cutoff)
            spatial_adj_list.append(temp_adata.uns['Spatial_Net'])
            # normalize the coordinate after KNN for numerical stability
            temp_adata = self.norm_coordinate(temp_adata)
            # update
            self.adata_list[i] = temp_adata
        # concat data
        self.adata = sc.concat(self.adata_list, merge="same")
        self.adata.obs['batches'] = self.adata.obs['batches'].astype('category')
        self.adata.obs['cell_ID'] = np.arange(0, self.adata.shape[0])
        # add Nat Info into unite anndata
        nat_info = pd.concat(spatial_adj_list)
        self.adata.uns['Spatial_Net'] = nat_info
        # convert cell_name into cell_id
        name_to_id = dict(zip(self.adata.obs.index, range(self.adata.shape[0])))
        self.adata.uns['Spatial_Net']['Cell1_ID'] = self.adata.uns['Spatial_Net']['Cell1_Name'].map(name_to_id)
        self.adata.uns['Spatial_Net']['Cell2_ID'] = self.adata.uns['Spatial_Net']['Cell2_Name'].map(name_to_id)
        self.re_dim()

    def re_dim(self):
        prune = self.adata.shape[1] - self.adata.shape[1] % self.args.heads
        self.adata = self.adata[:, 0: prune]

    @staticmethod
    def norm_coordinate(adata):
        loc = adata.obsm['spatial']
        # 0-1 norm
        normalized_coords = (loc - loc.min(0)) / (loc.max(0) - loc.min(0))
        adata.obsm['original_spatial'] = adata.obsm['spatial']
        # (0,0) as center
        adata.obsm['spatial'] = normalized_coords - 0.5
        # sc.pl.spatial(adata, img_key="hires", color=["Region"], spot_size=0.01)
        adata.obs['loc_x'] = adata.obsm['spatial'][:, 0]
        adata.obs['loc_y'] = adata.obsm['spatial'][:, 1]
        return adata

    def create_pyg_data(self):
        row_and_col = torch.LongTensor(np.array([self.adata.uns['Spatial_Net']['Cell1_ID'].to_numpy(),
                                                 self.adata.uns['Spatial_Net']['Cell2_ID'].to_numpy()]))
        # one weight
        edge_value = torch.FloatTensor(np.ones(self.adata.uns['Spatial_Net'].shape[0]))
        # contain self-loop
        try:
            data = Data(x=torch.as_tensor(self.adata.X.todense(), dtype=torch.float32),  # Node feature matrix
                        y=torch.tensor(self.adata.obs[['cell_ID', 'batches', 'loc_x', 'loc_y']].to_numpy()),
                        # Graph-level or node-level ground-truth labels
                        edge_index=row_and_col,  # Graph connectivity in COO format
                        edge_attr=edge_value,  # Edge feature matrix
                        )
        except AttributeError:
            data = Data(x=torch.as_tensor(self.adata.X, dtype=torch.float32),  # Node feature matrix
                        y=torch.tensor(self.adata.obs[['cell_ID', 'batches', 'loc_x', 'loc_y']].to_numpy()),
                        # Graph-level or node-level ground-truth labels
                        edge_index=row_and_col,  # Graph connectivity in COO format
                        edge_attr=edge_value,  # Edge feature matrix
                        )
        if data.validate(raise_on_error=True):
            print('The PyG data u create is qualified')
        self.data = data
        print('=The graph contains %d edges, %d cells=' % (self.data.edge_index.shape[1], self.data.x.shape[0]))
        # to show that if the graph contain enough biological information
        print('=   %.4f neighbors per cell on average   =' % (self.data.edge_index.shape[1] / self.data.x.shape[0]))

    def generate_minibatch(self, loader_type, num_workers=5):
        self.create_pyg_data()
        if loader_type == 'RandomNodeLoader':
            # self.loader = RandomNodeLoader(self.data, num_parts=self.args.heads,
            #                                shuffle=True, num_workers=num_workers)
            self.loader = RandomNodeLoader(self.data, num_parts=1, shuffle=False, num_workers=num_workers)

        elif loader_type == 'ClusterDataLoader':
            # num_parts: The number of partitions
            # batch_size: The number of partitions within a batch
            # so batch_num = num_parts/batch_size (args.heads)
            self.loader = ClusterLoader(ClusterData(self.data, num_parts=self.args.heads * 32),
                                        batch_size=32, shuffle=True)
        print('=              subgraph Info             =')
        print("============================================")

        i = 0
        for batch in self.loader:
            print('=           Batch {}: {} nodes           ='.format(i, batch.x.shape[0]))
            # unique_values, counts = torch.unique(batch.y[:, 1], return_counts=True)
            # result_dict = dict(zip(unique_values.tolist(), counts.tolist()))
            # print('node_num of slice1:{}, node_num of slice2:{}'.format(result_dict[1],result_dict[2]) )
            print('=   %.4f neighbors per cell on average   =' % (batch.edge_index.shape[1] / batch.x.shape[0] - 1))
            print('batch:{}, node num:{}'.format(torch.unique(batch.y[:, 1], return_counts=True)[0].numpy(),
                                                 torch.unique(batch.y[:, 1], return_counts=True)[1].numpy()))
            i = i + 1
            print("============================================")
