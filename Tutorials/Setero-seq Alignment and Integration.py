import sys
import time
from spatialFuser import *
import scanpy as sc
import pandas as pd
import numpy as np

sys.path.append("..")

# load args:
print("============================================")
print("=              Setting Params              =")
slice_1_args = def_training_args()
slice_1_args.hidden = [512, 32]
slice_1_args.epochs = 500
slice_1_args.lr = 3e-4
slice_1_args.K = 4
slice_1_args.heads = 1
slice_1_args.alpha = 0

slice_2_args = def_training_args()
slice_2_args.hidden = [512, 32]
slice_2_args.epochs = 500
slice_2_args.lr = 3e-4
slice_2_args.K = 4
slice_2_args.heads = 1
slice_2_args.alpha = 0

integration_args = def_training_args()
integration_args.hidden = [32, 32]
integration_args.fusion_epoch = 200
integration_args.lr = 5e-3
integration_args.match_step_size = 10
integration_args.tau = 0.1
integration_args.roi_radius = 0.15
integration_args.epsilon = 1
integration_args.m_top_K = 2
integration_args.beta_rec = 60
integration_args.beta_dir = 1

# load data:
print("============================================")
print("=               Loading Data               =")
slice_1_dataLoader = SpatialFuserDataLoader(slice_1_args,
                                            data_dir='/public8/lilab/student/whcai/Integration/data/Stereo-seq_Axolotl_Brain',
                                            data_tech='seq-based',
                                            files=['Stereo-seq_wei2022single_Stage54_data.h5ad'])
slice_1_dataLoader.load_adata()
slice_1_dataLoader.pre_processing(n_svgs=3000, k_cutoff=slice_1_args.K, batch_label=[1])

slice_2_dataLoader = SpatialFuserDataLoader(slice_2_args,
                                            data_dir='/public8/lilab/student/whcai/Integration/data/Stereo-seq_Axolotl_Brain',
                                            data_tech='seq-based',
                                            files=['Stereo-seq_wei2022single_Stage57_data.h5ad'])
slice_2_dataLoader.load_adata()
slice_2_dataLoader.pre_processing(n_svgs=3000, k_cutoff=slice_2_args.K, batch_label=[2])

highly_variable_1 = slice_1_dataLoader.adata.var[slice_1_dataLoader.adata.var['highly_variable']].index
highly_variable_2 = slice_2_dataLoader.adata.var[slice_2_dataLoader.adata.var['highly_variable']].index

# find same SVGs
intersection_genes = np.intersect1d(highly_variable_1, highly_variable_2)
slice_1_dataLoader.adata.var['inter_SVGs'] = False
slice_1_dataLoader.adata.var.loc[intersection_genes, 'inter_SVGs'] = True
slice_2_dataLoader.adata.var['inter_SVGs'] = False
slice_2_dataLoader.adata.var.loc[intersection_genes, 'inter_SVGs'] = True
slice_1_dataLoader.adata = slice_1_dataLoader.adata[:, intersection_genes]
slice_2_dataLoader.adata = slice_2_dataLoader.adata[:, intersection_genes]

# anndata to PyG object
slice_1_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)
slice_1_dataLoader.adata.obs['Region'] = slice_1_dataLoader.adata.obs['Annotation']
slice_2_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)
slice_2_dataLoader.adata.obs['Region'] = slice_2_dataLoader.adata.obs['Annotation']


# train
print("============================================")
print("=              Begin to Train              =")
training_time = time.time()
[adata1, adata2], trainer = train_integration([slice_1_args, slice_2_args, integration_args],
                                              [slice_1_dataLoader, slice_2_dataLoader])
print("=            Training Finished!            =")
print("Total time elapsed: {:.4f}s".format(time.time() - training_time))
print("============================================")


# evaluate and plot
# spatial domain detection
adata1_leiden_result, adata1_louvain_result, adata1_mclust_result = metrics(adata1,
                                                                            save_loc='_Stereo-Seq_stage54.png',
                                                                            resolution=0.1,
                                                                            spot_size=0.02,
                                                                            cluster_label='Region',
                                                                            plot_color=["mclust"],
                                                                            mclust_model='EEE',
                                                                            embed_label='fused_embedding',
                                                                            vis=True,
                                                                            save=False)

adata2_leiden_result, adata2_louvain_result, adata2_mclust_result = metrics(adata2,
                                                                            save_loc='_Stereo-Seq_stage57.png',
                                                                            resolution=0.1,
                                                                            spot_size=0.02,
                                                                            cluster_label='Region',
                                                                            plot_color=["mclust"],
                                                                            mclust_model='EEE',
                                                                            embed_label='fused_embedding',
                                                                            vis=True,
                                                                            save=False)

# batch effect correction (integration)
checkBatch(adata1, adata2, save=None)

# show alignment
adata1_df = pd.DataFrame({'index': range(adata1.shape[0]),
                          'x': adata1.obsm['spatial'][:, 0],
                          'y': adata1.obsm['spatial'][:, 1],
                          'celltype': adata1.obs['Region']})
adata2_df = pd.DataFrame({'index': range(adata2.shape[0]),
                          'x': adata2.obsm['spatial'][:, 0],
                          'y': adata2.obsm['spatial'][:, 1],
                          'celltype': adata2.obs['Region']})
matching = np.array([trainer.match_in_adata1.data.cpu().numpy(), trainer.match_in_adata2.data.cpu().numpy()])
multi_align = match_3D_multi(adata1_df, adata2_df, matching, meta='celltype',
                             scale_coordinate=True, subsample_size=300, exchange_xy=False)
multi_align.draw_3D(target='all_type', size=[7, 8], line_width=1, point_size=[0.8, 0.8], line_color='blue',
                    hide_axis=True, show_error=False, only_show_correct=True, only_show_error=False,
                    line_alpha=0.5)

# evaluate alignment
valid_ratio, accuracy = all_matching(adata1, adata2, 0.95, 0.15, save_loc=None, file_name=None)
