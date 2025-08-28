import sys
import time
from spatialFuser import *
import scanpy as sc
sys.path.append("..")

# load args:
print("============================================")
print("=              Setting Params              =")
args = def_training_args()
args.epochs = 500
args.K = 10
args.step = 100
args.heads = 4
args.alpha = 0
args.hidden = [32]
args.lr = 8e-3 #8e-3/7e-3

# load data:
print("============================================")
print("=               Loading Data               =")
loadtime = time.time()
dataLoader = SpatialFuserDataLoader(args,
                                    data_dir='/public8/lilab/student/whcai/Integration/data/osmFISH_mouse_cortex',
                                    data_tech='image-based',
                                    files=['osmFISH_codeluppi2018spatial_cortex_data.h5ad'])
dataLoader.load_adata()
dataLoader.remove_type(obs_name='Region', target='Excluded')
dataLoader.pre_processing(n_svgs=3000, k_cutoff=args.K, batch_label=[1])
dataLoader.generate_minibatch(loader_type='RandomNodeLoader', num_workers=5)

# train
print("============================================")
print("=              Begin to Train              =")
training_time = time.time()
# Train model
adata, trainer, loss_list = train_emb(args, dataLoader)
print("=            Training Finished!            =")
print("Total time elapsed: {:.4f}s".format(time.time() - training_time))
print("============================================")

# show param number
total_params, total_trainable_params = show_para_num(trainer.model)
param_num = {'total_params': total_params,
             'trainable_params': total_trainable_params,
             'non-trainable_params': total_params - total_trainable_params}

# evaluate and plot
leiden_result, louvain_result, mclust_result = metrics(adata,
                                                       save_loc='_osmFISH.png',
                                                       resolution=0.1,
                                                       spot_size=0.015,
                                                       cluster_label='Region',
                                                       plot_color=["mclust"],
                                                       mclust_model='EEE',
                                                       embed_label='embedding',
                                                       vis=False,
                                                       save=False)
print(mclust_result)

sc.tl.umap(adata)
adata.obs['mclust'] = adata.obs['mclust'].astype(str)

region_color = {
    'Pia Layer 1': "#A1C4D9",
    'Layer 2-3 medial': "#D4A0A2",
    'Layer 2-3 lateral': "#A8C4A5",
    'Layer 3-4': "#746053",
    'Layer 4': "#8E7D73",
    'Layer 5': "#A8B86A",
    'Layer 6': "#B9A88D",
    'White matter': "#B5D0B4",
    'Internal Capsule Caudoputamen': "#6D8498",
    'Hippocampus': "#A57058",
    'Ventricle': "#E77A67",
    '1.0': "#B9A88D",
    '2.0': "#A8C4A5",
    '3.0': "#8E7D73",
    '4.0': "#E77A67",
    '5.0': "#A8B86A",
    '6.0': "#D4A0A2",
    '7.0': "#B5D0B4",
    '8.0': "#A57058",
    '9.0': "#A1C4D9",
    '10.0': "#6D8498",
    '11.0': "#746053",
}

sc.pl.spatial(adata, img_key="hires", color=['Region', "mclust"],
              spot_size=0.015,
              title=['Original',
                     'mclust ARI: {:.2f}'.format(mclust_result['ARI'][0])],
              wspace=0.23,
              palette=region_color,
              frameon=False,
              save='_osmFISH.png'
              )
