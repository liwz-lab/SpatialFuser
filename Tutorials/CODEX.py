import sys
import pandas as pd
import time
import numpy as np
from spatialFuser import *
import scanpy as sc
sys.path.append("..")


# load args:
print("============================================")
print("=              Setting Params              =")
args = def_training_args()
args.epochs = 500
args.K = 2
args.step = 100
args.heads = 4
args.hidden = [64, 32]
args.alpha = 0.6
args.lr = 7e-3  # 7e-3, 8e-3

# load data:
print("============================================")
print("=               Loading Data               =")
loadtime = time.time()
dataLoader = SpatialFuserDataLoader(args,
                                    data_dir='/public8/lilab/student/whcai/Integration/data/CODEX_bladderCancer/codex',
                                    data_tech='image-based',
                                    files=['codex_72_210308_mean_upperType.h5ad'])
dataLoader.load_adata()
dataLoader.pre_processing(n_svgs=3000, k_cutoff=args.K, batch_label=[1])
dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)

# train
print("============================================")
print("=              Begin to Train              =")
training_time = time.time()
adata, trainer, loss_list = train_emb(args, dataLoader)
print("=            Training Finished!            =")
print("Total time elapsed: {:.4f}s".format(time.time() - training_time))

# show param number
print("============================================")
show_para_num(trainer.model)
visualize_loss(loss_list)
# checkGlobalAtt(trainer)

# evaluate and plot
leiden_result, louvain_result, mclust_result = metrics(adata,
                                                       save_loc='_codex_72_210308_mean.png',
                                                       resolution=0.1,
                                                       spot_size=0.015,
                                                       cluster_label='Region',
                                                       plot_color=["mclust"],
                                                       mclust_model='EEE',
                                                       embed_label='embedding',
                                                       vis=False,
                                                       save=False)
print(mclust_result)

adata.obs['mclust'] = adata.obs['mclust'].astype(str)

for label in ['Epithelial', 'Immune', 'Stromal']:
    adata.obs[label] = np.where(
        adata.obs['Region'] == label,
        label,
        pd.NA
    )

for clust in ['1.0', '2.0', '3.0']:
    col_name = f'mclust_{clust}'
    adata.obs[col_name] = np.where(
        adata.obs['mclust'] == clust,
        clust,
        pd.NA
    )

category_cols = ['Epithelial', 'Immune', 'Stromal', 'mclust_1.0', 'mclust_2.0', 'mclust_3.0']
adata.obs[category_cols] = adata.obs[category_cols].apply(
    lambda x: x.astype('category'),
    axis=0
)

region_color = {
    'Epithelial': "#75436A",
    'Immune': "#9A80B5",
    'Stromal': "#A58D9F",
    '1.0': "#75436A",
    '2.0': "#9A80B5",
    '3.0': "#A58D9F",
}

sc.pl.spatial(adata,
              img_key="None",
              color=['Epithelial', "mclust_1.0"],
              spot_size=0.015,
              title=['Epithelial', 'Pred'],
              wspace=0.2,
              palette=region_color,
              frameon=False,
              save='_CODEX_Epithelial.png'
              )

sc.pl.spatial(adata,
              img_key="None",
              color=['Immune', "mclust_2.0"],
              spot_size=0.015,
              title=['Immune', 'Pred'],
              wspace=0.2,
              palette=region_color,
              frameon=False,
              save='_CODEX_Immune.png'
              )

sc.pl.spatial(adata,
              img_key="None",
              color=['Stromal', "mclust_3.0"],
              spot_size=0.015,
              title=['Stromal', 'Pred'],
              wspace=0.2,
              palette=region_color,
              frameon=False,
              save='_CODEX_Stromal.png'
              )