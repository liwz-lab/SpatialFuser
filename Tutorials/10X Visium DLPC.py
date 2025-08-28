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
args.K = 6
args.step = 100
args.heads = 4
args.alpha = 0
args.lr = 2e-3

# load data:
print("============================================")
print("=               Loading Data               =")
loadtime = time.time()
dataLoader = SpatialFuserDataLoader(args,
                                    data_dir='/public8/lilab/student/whcai/Integration/data/10X_DLPFC',
                                    data_tech='seq-based',
                                    files=['10X_Visium_maynard2021trans_151672_data.h5ad'])
dataLoader.load_adata()
# dataLoader.remove_nan(obs_name='Region')
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
                                                       save_loc='_151672.png',
                                                       resolution=0.1,
                                                       spot_size=0.015,
                                                       cluster_label='Region',
                                                       plot_color=["mclust"],
                                                       mclust_model='EEE',
                                                       embed_label='embedding',
                                                       vis=True,
                                                       save=True)
