import argparse
import torch
import pandas as pd
import sklearn.neighbors
import numpy as np


def def_training_args(show_detail=False):
    """
    Return all hyperparameter settings.

    Args:
        show_detail: bool. If True, display all hyperparameters during initialization (default is True).

    Returns:
        args: argparse.Namespace. Object containing all hyperparameter settings.
    """

    # Training settings
    parser = argparse.ArgumentParser()
    # dims of hidden layer
    parser.add_argument('--hidden', type=int, default=[512, 32],
                        help='dims of hidden layer.')
    # heads num
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of heads.')
    # whether to apply an two-layer MLP after every EncoderLayer.
    parser.add_argument('--encoder_mlp', type=bool, default=True,
                        help='whether to apply an two-layer MLP after every EncoderLayer.')
    # whether to apply an two-layer MLP after every DecoderLayer.
    parser.add_argument('--decoder_mlp', type=bool, default=False,
                        help='whether to apply an two-layer MLP after every DecoderLayer.')
    # The dimension of the hidden layer is how many times the dimension of the input layer.
    parser.add_argument('--mlp_hidden_times', type=int, default=1,
                        help='The dimension of the hidden layer is how many times the dimension of the input layer.')
    # whether to use nonlinear activation function
    parser.add_argument('--nonlinear', type=bool, default=True,
                        help='whether to use non-linear activation function.')
    # random seeds
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    # epoch
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    # learning rate
    parser.add_argument('--lr', type=float, default=4e-3,
                        help='Initial learning rate.')
    # K for cluster
    parser.add_argument('--K', type=float, default=10,
                        help='K for cluster.')
    # weights of long-range atts
    parser.add_argument('--alpha', type=int, default=0,
                        help='weights of long-range atts.')
    #  of long-range atts
    parser.add_argument('--step', type=int, default=100,
                        help='update long-range atts.')
    # weight of reconstruction loss
    parser.add_argument('--beta_rec', type=int, default=1,
                        help='weights of MSE in fusion loss.')
    # weight of cos-direction loss
    parser.add_argument('--beta_dir', type=int, default=1,
                        help='weights of cos-distance loss in fusion loss.')
    # epoch of the fusion process
    parser.add_argument('--fusion_epoch', type=int, default=500,
                        help='Number of epochs for the fusion process.')
    # temperature coefficient
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Temperature coefficient.')
    # how many steps to update high-confidence matches
    parser.add_argument('--match_step_size', type=int, default=10,
                        help='Number of steps to update high-confidence matches.')
    # radius for constructing matches
    parser.add_argument('--roi_radius', type=float, default=0.1,
                        help='Neighborhood radius for constructing matches.')
    # K for Mutual top-K
    parser.add_argument('--m_top_K', type=int, default=1,
                        help='K value for Mutual top-K.')
    # Sinkhorn parameter
    parser.add_argument('--epsilon', type=float, default=1,
                        help='Sinkhorn parameter.')
    # weight decay
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    # dropout
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability).')
    # The percentage of nodes to be masked
    parser.add_argument('--mask_rate', type=float, default=0,
                        help='The percentage of nodes to be masked.')
    # training details
    parser.add_argument('--test_multi_epochs', type=bool, default=False,
                        help='whether to show details of training.')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='whether to show details of training.')
    parser.add_argument('--check_global_att_index', type=int, default=7,
                        help='whether to show details of training.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    # for debug
    parser.add_argument('--mode', type=str, default='client',
                        help='run.')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='run.')
    parser.add_argument('--port', type=int, default=40148,
                        help='run.')

    # console
    # args = parser.parse_args()
    # Jupyter
    args = parser.parse_known_args()[0]

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # display hyperparameters
    if show_detail:
        print("============================================")
        show_Hyperparameter(args)
        print("============================================")
    return args


def show_Hyperparameter(args):
    argsDict = args.__dict__
    print(argsDict)
    print('the settings are as following')
    for key in argsDict:
        print(key, ':', argsDict[key])


def get_spatial_adj(adata, k_cutoff=10):
    """
    Construct the spatial neighbor networks.

    Args:
        adata: AnnData. The AnnData object from the scanpy package.
        k_cutoff: int. Number of nearest neighbors to use for constructing the spatial network.

    Returns:
        None. The spatial networks are saved in `adata.uns['Spatial_Net']`.
    """

    adata = adata
    print('=         Calculating spatial graph        =')
    # coor is positional information
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    NN_list = []
    for it in range(indices.shape[0]):
        NN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    # List to dataframe
    Spatial_Net = pd.concat(NN_list)
    Spatial_Net.columns = ['Cell1_Name', 'Cell2_Name', 'Distance']
    # with self-loop
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] >= 0,]
    # convert cell_ID into cell_name and save it
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1_Name'] = Spatial_Net['Cell1_Name'].map(id_cell_trans)
    Spatial_Net['Cell2_Name'] = Spatial_Net['Cell2_Name'].map(id_cell_trans)
    # save the Net information into anndata
    adata.uns['Spatial_Net'] = Spatial_Net
    # basic information of graph
    # print('=The graph contains %d edges, %d cells=' % (Spatial_Net.shape[0], adata.n_obs))
    # # to show that if the graph contain enough biological information
    # print('=   %.4f neighbors per cell on average   =' % (Spatial_Net.shape[0] / adata.n_obs))
    return adata


def generate_src_points(n_points, field_length):
    px = (np.random.rand(n_points) - 0.5) * field_length
    py = (np.random.rand(n_points) - 0.5) * field_length
    return np.vstack((px, py)).T
