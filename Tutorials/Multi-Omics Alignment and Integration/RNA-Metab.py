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
slice_1_args.lr = 3e-3
slice_1_args.K = 4
slice_1_args.heads = 4
slice_1_args.alpha = 0

slice_2_args = def_training_args()
slice_2_args.hidden = [32]
slice_2_args.epochs = 500
slice_2_args.lr = 5e-3
slice_2_args.K = 4
slice_2_args.heads = 4
slice_2_args.alpha = 0

integration_args = def_training_args()
integration_args.hidden = [32, 32]
integration_args.fusion_epoch = 200
integration_args.lr = 3e-3
integration_args.match_step_size = 20
integration_args.tau = 0.1
integration_args.roi_radius = 0.02
integration_args.epsilon = 1
integration_args.m_top_K = 2
integration_args.beta_rec = 50
integration_args.beta_dir = 0.1
integration_args.verbose = False

# load data:
print("============================================")
print("=               Loading Data               =")
slice_1_dataLoader = SpatialFuserDataLoader(slice_1_args,
                                            data_dir='/public8/lilab/student/whcai/Integration/data/cell_fangqing_zhao',
                                            data_tech='seq-based',
                                            files=['Cerebellum-MAGIC-seq_raw.h5ad'])
slice_1_dataLoader.load_adata()
slice_1_dataLoader.pre_processing(n_svgs=3000, k_cutoff=slice_1_args.K, batch_label=[1])
slice_1_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)

slice_2_dataLoader = SpatialFuserDataLoader(slice_2_args,
                                            data_dir='/public8/lilab/student/whcai/Integration/data/cell_fangqing_zhao',
                                            data_tech='image-based',
                                            files=['Cerebellum-MALDI-MSI_raw.h5ad'])
slice_2_dataLoader.load_adata()
slice_2_dataLoader.pre_processing(n_svgs=3000, k_cutoff=slice_2_args.K, batch_label=[2])
slice_2_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)

# anndata to PyG object
slice_1_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)
slice_1_dataLoader.adata.obs['Region'] = slice_1_dataLoader.adata.obs['cluster']
slice_2_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)
slice_2_dataLoader.adata.obs['Region'] = slice_2_dataLoader.adata.obs['cluster']

# pre-matching
slice_1_dataLoader, slice_2_dataLoader = ndt_pre_match(slice_1_dataLoader, slice_2_dataLoader)

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
                                                                            save_loc='_slice1.png',
                                                                            n_neighbors=12,
                                                                            resolution=0.1,
                                                                            spot_size=0.02,
                                                                            cluster_label='Region',
                                                                            plot_color=["louvain", ],
                                                                            mclust_model='EEE',
                                                                            embed_label='fused_embedding',
                                                                            vis=False,
                                                                            save=False)

adata2_leiden_result, adata2_louvain_result, adata2_mclust_result = metrics(adata2,
                                                                            save_loc='_slice2.png',
                                                                            n_neighbors=15,
                                                                            resolution=0.1,
                                                                            spot_size=0.015,
                                                                            cluster_label='Region',
                                                                            plot_color=["louvain"],
                                                                            mclust_model='EEE',
                                                                            embed_label='fused_embedding',
                                                                            vis=False,
                                                                            save=False)

region_color = {'Fiber tracts': "#a03b32",
                'Granular layer': "#ffcf4f",
                'Lateral recess': "#33658A",
                'Molecular layer': "#878bb4",

                '0': "#878bb4",
                '1': "#ffcf4f",
                '2': "#a03b32",
                '3': "#33658A",
                1: "#89606a",
                2: "#71a2b6",
                }

sc.pl.spatial(adata1,
              img_key=None,
              color=["Region", "louvain"],
              spot_size=0.02,
              title=['Original Annotation', 'Louvain ARI: {:.2f}'.format(adata1_louvain_result['ARI'][0])],
              wspace=0.1,
              palette=region_color,
              frameon=False,
              save='_RNA_Metab_RNA.png'
              )

sc.pl.spatial(adata2,
              img_key=None,
              color=["Region", "louvain"],
              spot_size=0.015,
              title=['Original Annotation', 'Louvain ARI: {:.2f}'.format(adata2_louvain_result['ARI'][0])],
              wspace=0.1,
              palette=region_color,
              frameon=False,
              save='_RNA_Metab_Metab.png'
              )

# modality bias correction (integration)
adata = sc.AnnData(X=np.concatenate([adata1.obsm['fused_embedding'], adata2.obsm['fused_embedding']]))
adata.obsm['spatial'] = np.concatenate([adata1.obsm['spatial'], adata2.obsm['spatial']])
adata.obs = pd.concat([adata1.obs, adata2.obs])
adata.obs['batches'] = adata.obs['batches'].astype('category')
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
sc.tl.umap(adata, min_dist=0.5, spread=1)
sc.pl.umap(adata,
           color=["Region", "batches"],
           wspace=0.2,
           palette=region_color,
           frameon=False,
           save='_RNA_Metab_check_batch.png')

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
                    line_alpha=0.5, save=None)

# evaluate alignment
valid_ratio, accuracy = all_matching(adata1, adata2,
                                     0.95,
                                     0.02,
                                     save_loc='./figures/',
                                     file_name='RNA_Metab')



