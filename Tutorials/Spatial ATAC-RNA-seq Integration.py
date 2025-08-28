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
slice_1_args.lr = 5e-4
slice_1_args.K = 4
slice_1_args.heads = 2
slice_1_args.alpha = 0

slice_2_args = def_training_args()
slice_2_args.hidden = [512, 32]
slice_2_args.epochs = 500
slice_2_args.lr = 1e-3
slice_2_args.K = 4
slice_2_args.heads = 2
slice_2_args.alpha = 0

integration_args = def_training_args()
integration_args.hidden = [32, 32]
integration_args.fusion_epoch = 500
integration_args.lr = 1e-4
integration_args.match_step_size = 20
integration_args.tau = 0.1
integration_args.roi_radius = 0.01
integration_args.epsilon = 1
integration_args.m_top_K = 2
integration_args.beta_rec = 50
integration_args.beta_dir = 0.1
integration_args.verbose = False

# load data:
print("============================================")
print("=               Loading Data               =")
slice_1_dataLoader = SpatialFuserDataLoader(slice_1_args,
                                            data_dir='/public8/lilab/student/whcai/Integration/data/Spatial-ATAC-RNA-seq_FanRong/MouseE13',
                                            data_tech='seq-based',
                                            files=['ATAC.h5ad'])

slice_1_dataLoader.load_adata()
slice_1_dataLoader.pre_processing(n_svgs=3000, k_cutoff=slice_1_args.K, batch_label=[1])
slice_1_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)

slice_2_dataLoader = SpatialFuserDataLoader(slice_2_args,
                                            data_dir='/public8/lilab/student/whcai/Integration/data/Spatial-ATAC-RNA-seq_FanRong/MouseE13',
                                            data_tech='seq-based',
                                            files=['RNA.h5ad'])
slice_2_dataLoader.load_adata()
slice_2_dataLoader.pre_processing(n_svgs=3000, k_cutoff=slice_2_args.K, batch_label=[2])
slice_2_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)

# annadata to PyG object
slice_1_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)
slice_1_dataLoader.adata.obs['Region'] = slice_1_dataLoader.adata.obs['Joint_clusters']
slice_2_dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)
slice_2_dataLoader.adata.obs['Region'] = slice_2_dataLoader.adata.obs['Joint_clusters']

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
sc.pp.neighbors(adata1, n_neighbors=10, use_rep='fused_embedding')
sc.tl.leiden(adata1, resolution=0.4)
sc.pl.spatial(adata1,
              img_key=None,
              color=["Region", "leiden"],
              spot_size=0.02,
              title=['Mannul Annotation', 'ATAC leiden'],
              wspace=0.1,
              frameon=False,
              save='_ATAC.png'
              )
sc.pp.neighbors(adata1, n_neighbors=10, use_rep='fused_embedding')
sc.tl.umap(adata1, min_dist=0.5, spread=1)
sc.pl.umap(adata1,
           color=["leiden"],
           wspace=0.2,
           frameon=False,
           title=['ATAC Umap'],
           legend_loc='on data',
           save='_ATAC.png')

sc.pp.neighbors(adata2, n_neighbors=10, use_rep='fused_embedding')
sc.tl.leiden(adata2, resolution=0.4)
sc.pl.spatial(adata2,
              img_key=None,
              color=["Region", "leiden"],
              spot_size=0.02,
              title=['Mannul Annotation', 'leiden', 'RNA leiden'],
              wspace=0.1,
              frameon=False,
              save='_RNA.png'
              )
sc.pp.neighbors(adata2, n_neighbors=10, use_rep='fused_embedding')
sc.tl.umap(adata2, min_dist=0.5, spread=1)
sc.pl.umap(adata2,
           color=["leiden"],
           wspace=0.2,
           frameon=False,
           title=['RNA Umap'],
           legend_loc='on data',
           save='_RNA.png')

# batch effect correction (integration)
region_color = {
    1: "#89606a",
    2: "#71a2b6",
}
adata = sc.AnnData(X=np.concatenate([adata1.obsm['fused_embedding'], adata2.obsm['fused_embedding']]))
adata.obsm['spatial'] = np.concatenate([adata1.obsm['spatial'], adata2.obsm['spatial']])
adata.obs = pd.concat([adata1.obs, adata2.obs])
adata.obs['batches'] = adata.obs['batches'].astype('category')
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
sc.tl.umap(adata, min_dist=0.5, spread=1)
sc.pl.umap(adata,
           color=["batches"],
           wspace=0.2,
           palette=region_color,
           frameon=False,
           save='_ATAC_RNA_check_batch.png')



