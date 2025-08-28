import random
import numpy as np
import scanpy as sc
import torch
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, \
    v_measure_score
from .loss import reconstruction_loss, overall_direction_loss
from .embModel import MCGATE
from .machingLayer import OptimalMatchingLayer
from .fusionLayer import fusionLayer
from .enval import mclust_R


class emb_trainer():
    """
    Trainer of the MCGATE model.

    Args:
        args: argparse.Namespace. All hyperparameter settings.
        loader: object. Data loader from `SpatialFuserDataLoader.loader`.
        adata: AnnData. AnnData object from `SpatialFuserDataLoader.dataLoader.adata`.
    """

    def __init__(self, args, loader, adata):
        self.args = args
        self.loss_list = []
        self.adata = adata
        for batch in loader:
            if self.args.heads > 1:
                self.input = batch.x
            else:
                self.input = batch.x.to_sparse_coo()
            self.spatial_adj = torch.sparse_coo_tensor(batch.edge_index, batch.edge_attr,
                                                        torch.Size([batch.x.shape[0],
                                                                    batch.x.shape[0]]))
            self.loc = batch.y[:, [2, 3]]

        self.model = MCGATE(self.args)

        if args.cuda:
            self.input = self.input.cuda()
            self.spatial_adj = self.spatial_adj.cuda()
            self.model.cuda()
            self.loc = self.loc.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.batch_size = self.input.shape[0]

        self.loss_list = []

        self.result_list = []
        self.globle_att = np.array([])

    def __call__(self):
        pbar = tqdm(range(self.args.epochs), ncols=120, position=0)
        for epoch in pbar:
            self.__run_epoch(epoch + 1)
            pbar.set_description('Epoch' + str(epoch + 1) + ' || '
                                 + 'loss: ' + str(self.loss.item().__round__(4)) + ' || '
                                 + 'MNN num: '
                                 + str(self.model.mnn_count.data.cpu().numpy())
                                 )

    def __run_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        self.embedding, self.X_decode = self.model.forward(self.spatial_adj, self.input)

        self.loss = reconstruction_loss(self.input, self.X_decode)
        self.loss.backward()

        self.optimizer.step()
        if self.args.cuda:
            torch.cuda.empty_cache()

        self.loss_list.append(self.loss.item())

        if (epoch % self.args.step == 0) & (epoch > 1) & (self.args.alpha > 0):
            if type(self.args.check_global_att_index) == list:
                temp_global_connection = \
                np.where(self.model.long_range_patten.to_dense()[self.args.check_global_att_index, :].data.cpu().numpy() != 0)[1]
                self.globle_att = np.append(self.globle_att, temp_global_connection).astype(int)
            elif type(self.args.check_global_att_index) == int:
                temp_global_connection = \
                np.where(self.model.long_range_patten.to_dense()[self.args.check_global_att_index, :].data.cpu().numpy() != 0)[0]
                self.globle_att = np.append(self.globle_att, temp_global_connection).astype(int)
            else:
                print('check_global_att_index should be int or list')

        if self.args.test_multi_epochs:
            if epoch % 100 == 0 and epoch != 1:
                resolution = 0.1
                mcluadata1_model = 'EEE'
                embed_label = 'embedding'
                self.adata.obsm[embed_label] = self.embedding.cpu().data.numpy()
                sc.pp.neighbors(self.adata, n_neighbors=10, use_rep=embed_label)
                sc.tl.leiden(self.adata, resolution=resolution)
                sc.tl.louvain(self.adata, resolution=resolution)
                self.adata = mclust_R(self.adata, modelNames=mcluadata1_model, used_obsm=embed_label,
                                      num_cluster=self.adata.obs.dropna(axis=0)['Region'].unique().shape[0])
                obs_df = self.adata.obs.dropna(axis=0)
                temp_result = pd.DataFrame(index=['leiden', 'louvain', 'mclust'],
                                           columns=['ARI', 'AMI', 'Homogeneity', 'Completeness', 'V_Measure'])

                for cluadata1_method in ['leiden', 'louvain', 'mclust']:
                    temp_result.loc[cluadata1_method, 'ARI'] = adjusted_rand_score(obs_df['Region'], obs_df[cluadata1_method])
                    temp_result.loc[cluadata1_method, 'AMI'] = adjusted_mutual_info_score(obs_df['Region'],
                                                                                      obs_df[cluadata1_method])
                    temp_result.loc[cluadata1_method, 'Homogeneity'] = homogeneity_score(obs_df['Region'],
                                                                                     obs_df[cluadata1_method])
                    temp_result.loc[cluadata1_method, 'Completeness'] = completeness_score(obs_df['Region'],
                                                                                       obs_df[cluadata1_method])
                    temp_result.loc[cluadata1_method, 'V_Measure'] = v_measure_score(obs_df['Region'], obs_df[cluadata1_method])

                header = pd.DataFrame(['========', '========', '========', '========', '========']).T
                header.index = ['{}'.format(epoch)]
                header.columns = temp_result.columns
                self.result_list.append(pd.concat([header, temp_result]))

                if self.args.verbose:
                    if epoch % 100 == 0:
                        sc.tl.umap(self.adata)
                        sc.pl.umap(self.adata, color=["Region", "leiden", "louvain", "mclust"],
                                   title=['{}'.format(epoch)])
                        sc.pl.spatial(self.adata, img_key="hires", color=["Region", "leiden", "louvain", "mclust"],
                                      title=['{}'.format(epoch)], spot_size = 0.015)

    def infer(self):
        return self.input, self.embedding, self.X_decode, self.loss_list


class integration_trainer():
    """
    Trainer of the entire SpatialFuser framework, including slice-specific MCGATEs, the matching layer, and the fusion layer.

    Args:
        args_list: list of argparse.Namespace. List of all hyperparameter settings for MCGATEs and the integration layers.
        dataLoader_list: list of SpatialFuserDataLoader. List of SpatialFuserDataLoader instances for each slice.
    """

    def __init__(self, args_list, dataLoader_list):
        self.args_list = args_list
        self.loss_list = []
        self.meta_adata1 = dataLoader_list[0].adata.obs
        self.meta_adata2 = dataLoader_list[1].adata.obs

        # load data
        for batch_in_data1 in dataLoader_list[0].loader:
            self.adata1_input = batch_in_data1.x
            self.adata1_adata2atial_adj = torch.sparse.FloatTensor(batch_in_data1.edge_index,
                                                               batch_in_data1.edge_attr,
                                                               torch.Size([batch_in_data1.x.shape[0],
                                                                           batch_in_data1.x.shape[0]])).to_dense()
            self.adata1_label = batch_in_data1.y[:, 1]
            self.adata1_loc = batch_in_data1.y[:, [2, 3]]

        for batch_in_data2 in dataLoader_list[1].loader:
            self.adata2_input = batch_in_data2.x
            self.adata2_adata2atial_adj = torch.sparse.FloatTensor(batch_in_data2.edge_index,
                                                               batch_in_data2.edge_attr,
                                                               torch.Size([batch_in_data2.x.shape[0],
                                                                           batch_in_data2.x.shape[0]])).to_dense()
            self.adata2_label = batch_in_data2.y[:, 1]
            self.adata2_loc = batch_in_data2.y[:, [2, 3]]

        self.all_label = torch.concat([self.adata1_label, self.adata2_label])

        # graph auto encoder
        self.adata1_MCGATE = MCGATE(self.args_list[0])
        self.adata2_MCGATE = MCGATE(self.args_list[1])

        # self.fusionLayer = DeepCCA(self.args_list[2].hidden)
        self.fusionLayer = fusionLayer(self.args_list[2].hidden)

        self.matchLayer = OptimalMatchingLayer(self.args_list[2])

        if args_list[0].cuda:
            self.adata1_input = self.adata1_input.cuda()
            self.adata1_adata2atial_adj = self.adata1_adata2atial_adj.cuda()
            self.adata1_loc = self.adata1_loc.cuda()

            self.adata2_input = self.adata2_input.cuda()
            self.adata2_adata2atial_adj = self.adata2_adata2atial_adj.cuda()
            self.adata2_loc = self.adata2_loc.cuda()

            self.all_label = self.all_label.cuda()

            self.adata1_MCGATE.cuda()
            self.adata2_MCGATE.cuda()

            self.matchLayer.cuda()
            self.fusionLayer.cuda()

        self.adata1_optimizer = optim.Adam(self.adata1_MCGATE.parameters(),
                                       lr=self.args_list[0].lr,
                                       weight_decay=self.args_list[0].weight_decay)

        self.adata2_optimizer = optim.Adam(self.adata2_MCGATE.parameters(),
                                       lr=self.args_list[1].lr,
                                       weight_decay=self.args_list[1].weight_decay)

        self.fusion_optimizer = optim.Adam(self.fusionLayer.parameters(),
                                           lr=self.args_list[2].lr,
                                           weight_decay=self.args_list[2].weight_decay)

        self.adata1_embedding_loss_list = []
        self.adata2_embedding_loss_list = []
        
        self.Fusion_loss_list = []
        self.Dir_loss_list = []
        self.MSE_loss_list = []

    def pretrain(self):
        pbar = tqdm(range(self.args_list[0].epochs), ncols=120, position=0)
        for epoch in pbar:

            self.pretrain_embd()
            pbar.set_description('Epoch ' + str(epoch + 1) + ' || '
                                 + 'adata1_pretrain_loss: ' + str(self.adata1_pretrain_loss.item().__round__(4)) + ' || '
                                 + 'adata2_pretrain_loss: ' + str(self.adata2_pretrain_loss.item().__round__(4)) + ' || '
                                 )

    def pretrain_embd(self):
        self.adata1_MCGATE.train()
        self.adata1_optimizer.zero_grad()

        self.adata2_MCGATE.train()
        self.adata2_optimizer.zero_grad()

        self.fusionLayer.train()
        self.fusion_optimizer.zero_grad()

        # encode
        self.adata1_MCGATE.embedding, _ = self.adata1_MCGATE.forward(self.adata1_adata2atial_adj.detach(), self.adata1_input.detach())

        self.adata2_MCGATE.embedding, _ = self.adata2_MCGATE.forward(self.adata2_adata2atial_adj.detach(), self.adata2_input.detach())

        # concat
        embd = torch.concat([self.adata1_MCGATE.embedding, self.adata2_MCGATE.embedding], dim=0)

        # shuffle
        perm = torch.randperm(embd.shape[0])
        shuffled_embd = embd[perm]
        fused = self.fusionLayer.forward(shuffled_embd)

        # reindex
        inverse_perm = torch.argsort(perm)
        restored_fused = fused[inverse_perm]

        # split
        self.adata1_embedding = restored_fused[0:self.adata1_MCGATE.embedding.shape[0], :]
        self.adata2_embedding = restored_fused[self.adata1_MCGATE.embedding.shape[0]:, :]

        # decode
        self.adata1_output = self.adata1_MCGATE.decoder.forward(self.adata1_embedding)
        self.adata2_output = self.adata2_MCGATE.decoder.forward(self.adata2_embedding)

        self.adata1_pretrain_loss = reconstruction_loss(self.adata1_input.detach(), self.adata1_output)
        self.adata2_pretrain_loss = reconstruction_loss(self.adata2_input.detach(), self.adata2_output)
        # self.fusion_loss = reconstruction_loss(torch.concat([self.adata1_input.detach(),self.adata2_input.detach()]),
        #                                        torch.concat([self.adata1_output,self.adata2_output]))

        total_loss = self.adata1_pretrain_loss + self.adata2_pretrain_loss

        total_loss.backward()

        # backward
        self.adata1_optimizer.step()
        self.adata2_optimizer.step()
        self.fusion_optimizer.step()

        self.adata1_embedding_loss_list.append(self.adata1_pretrain_loss.item())
        self.adata2_embedding_loss_list.append(self.adata2_pretrain_loss.item())

        torch.cuda.empty_cache()

    def fusion(self):
        pbar = tqdm(range(self.args_list[2].fusion_epoch), ncols=120, position=0)
        for epoch in pbar:
            self.fusion_epoch(epoch)
            pbar.set_description('Epoch ' + str(epoch + 1) + ' || '
                                 + 'Fusion_loss : ' + str(self.Fusion_loss.item().__round__(4)) + ' || '
                                 + 'MSE_loss: ' + str(self.MSE_loss.item().__round__(4)) + ' || '
                                 + 'Dir_loss: ' + str(self.Dir_loss.item().__round__(4)) + ' || '
                                 )

    def fusion_epoch(self, epoch):
        self.adata1_MCGATE.train()
        self.adata1_optimizer.zero_grad()

        self.adata2_MCGATE.train()
        self.adata2_optimizer.zero_grad()

        self.fusionLayer.train()
        self.fusion_optimizer.zero_grad()

        # encode
        self.adata1_MCGATE.identify_sparse_spatial_patten(self.adata1_adata2atial_adj.detach())
        self.adata1_MCGATE.embedding = self.adata1_MCGATE.encoder.forward(self.adata1_MCGATE.local_patten,
                                                            self.adata1_MCGATE.long_range_patten,
                                                            self.adata1_input.detach())

        self.adata2_MCGATE.identify_sparse_spatial_patten(self.adata2_adata2atial_adj.detach())
        self.adata2_MCGATE.embedding = self.adata2_MCGATE.encoder.forward(self.adata2_MCGATE.local_patten,
                                                            self.adata2_MCGATE.long_range_patten,
                                                            self.adata2_input.detach())

        # concat
        embd = torch.concat([self.adata1_MCGATE.embedding, self.adata2_MCGATE.embedding], dim=0)

        # shuffle
        perm = torch.randperm(embd.shape[0])
        shuffled_embd = embd[perm]

        # fuse
        fused = self.fusionLayer.forward(shuffled_embd)

        # reindex
        inverse_perm = torch.argsort(perm)
        restored_fused = fused[inverse_perm]

        # split
        self.adata1_fused = restored_fused[0:self.adata1_MCGATE.embedding.shape[0], :]
        self.adata2_fused = restored_fused[self.adata1_MCGATE.embedding.shape[0]:, :]

        # decode
        self.adata1_output = self.adata1_MCGATE.decoder.forward(self.adata1_fused)
        self.adata2_output = self.adata2_MCGATE.decoder.forward(self.adata2_fused)

        if epoch % self.args_list[2].match_step_size == 0:
            # get anchor-positive and anchor-negative pairs
            self.match_in_adata1, self.match_in_adata2,\
            self.negative_in_adata1, self.negative_in_adata2 = self.matchLayer.forward(self.adata1_fused, self.adata2_fused,
                                                                               self.adata1_loc, self.adata2_loc,
                                                                               100)

            if self.args_list[2].verbose:
                acc = self.checkMatchingACC()
                print('acc:', acc)
                print('n_matching:', self.match_in_adata2.shape[0])

        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
        # improved triplet_loss
        self.Fusion_loss = torch.pairwise_distance(self.adata1_fused[self.match_in_adata1, :],
                                                   self.adata2_fused[self.match_in_adata2, :]).mean()\
                           + triplet_loss(self.adata1_fused[self.match_in_adata1, :],
                                          self.adata2_fused[self.match_in_adata2, :],
                                          self.adata1_fused[self.negative_in_adata1, :])\
                           + triplet_loss(self.adata2_fused[self.match_in_adata2, :],
                                          self.adata1_fused[self.match_in_adata1, :],
                                          self.adata2_fused[self.negative_in_adata2, :])

        # use reconstruction loss to supervise fusion process
        self.adata1_recon_loss = reconstruction_loss(self.adata1_input, self.adata1_output)
        self.adata2_recon_loss = reconstruction_loss(self.adata2_input, self.adata2_output)
        self.MSE_loss = self.adata1_recon_loss + self.adata2_recon_loss

        # direction control loss
        self.Dir_loss = overall_direction_loss(self.adata1_fused, self.adata2_fused)

        # all loss
        loss = self.Fusion_loss + self.args_list[2].beta_rec * self.MSE_loss + self.args_list[2].beta_dir * self.Dir_loss
        loss.backward()

        self.adata1_optimizer.step()
        self.adata2_optimizer.step()
        self.fusion_optimizer.step()

        self.Fusion_loss_list.append(self.Fusion_loss.item())
        self.Dir_loss_list.append(self.Dir_loss.item())
        self.MSE_loss_list.append(self.MSE_loss.item())
        torch.cuda.empty_cache()

    def checkMatchingACC(self):
        match_in_adata1 = self.match_in_adata1.numpy()
        match_in_adata2 = self.match_in_adata2.numpy()

        cell_class_adata1 = self.meta_adata1.iloc[match_in_adata1]['Region'].astype(str).values
        cell_class_adata2 = self.meta_adata2.iloc[match_in_adata2]['Region'].astype(str).values

        correct = (cell_class_adata1 == cell_class_adata2).sum()
        all_match_num = len(match_in_adata1)
        acc = correct / all_match_num
        return acc

    def infer(self):
        return [self.adata1_input, self.adata2_input], [self.adata1_embedding, self.adata2_embedding], \
            [self.adata1_output, self.adata2_output], self.adata1_embedding_loss_list, self.adata2_embedding_loss_list, self.Fusion_loss_list


def train_emb(args, dataLoader):
    """
    Train the MCGATE model.

    Args:
        args: Namespace. All hyperparameter and training settings.
        dataLoader: spatialFuserDataLoader. Data loader containing the input spatial multi-omics dataset.

    Returns:
        adata: The processed AnnData object with updated results.
        trainer: The trainer instance used for model optimization.
        loss_list: Training loss values recorded across epochs.
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.hidden = [dataLoader.data.x.shape[1]] + args.hidden

    trainer = emb_trainer(args, dataLoader.loader, dataLoader.adata)
    # train
    trainer()
    original, embedding, X_decode, loss_list = trainer.infer()

    # saving results
    dataLoader.adata.uns['MCGATE_loss'] = loss_list

    dataLoader.adata.obsm['embedding'] = embedding.cpu().data.numpy()

    dataLoader.adata.obsm['ReX'] = X_decode.cpu().data.numpy()

    return dataLoader.adata, trainer, loss_list


def train_integration(args_list, dataLoader_list):
    """
    Train the entire SpatialFuser framework, including slice-specific MCGATEs, the matching layer, and the fusion layer.

    Args:
        args_list: list. A list of argument objects for each slice-specific MCGATE model and the alignment and integration process.
        dataLoader_list: list. A list of SpatialFuserDataLoader instances for each slice.

    Returns:
        dataLoader_list: A list containing the updated AnnData objects
        trainer: The trainer instance after training.
    """

    np.random.seed(args_list[0].seed)
    torch.manual_seed(args_list[0].seed)
    random.seed(args_list[0].seed)
    if args_list[0].cuda:
        torch.cuda.manual_seed(args_list[0].seed)

    for i in range(2):
        args_list[i].hidden = [dataLoader_list[i].data.x.shape[1]] + args_list[i].hidden

    # if args.alpha == 0:
    trainer = integration_trainer(args_list, dataLoader_list)
    # train
    trainer.pretrain()
    trainer.fusion()
    original_list, embedding_list, X_decode_list, adata1_embedding_loss_list, adata2_embedding_loss_list, D_loss_list = trainer.infer()

    # saving results
    dataLoader_list[0].adata.uns['adata1_embedding_loss_list'] = trainer.adata1_embedding_loss_list
    dataLoader_list[1].adata.uns['adata1_embedding_loss_list'] = trainer.adata1_embedding_loss_list

    dataLoader_list[0].adata.uns['adata2_embedding_loss_list'] = trainer.adata2_embedding_loss_list
    dataLoader_list[1].adata.uns['adata2_embedding_loss_list'] = trainer.adata2_embedding_loss_list

    dataLoader_list[0].adata.uns['D_loss_list'] = trainer.Fusion_loss_list
    dataLoader_list[1].adata.uns['D_loss_list'] = trainer.Fusion_loss_list

    dataLoader_list[0].adata.obsm['embedding'] = trainer.adata1_embedding.cpu().data.numpy()
    dataLoader_list[1].adata.obsm['embedding'] = trainer.adata2_embedding.cpu().data.numpy()

    dataLoader_list[0].adata.obsm['ReX'] = trainer.adata1_output.cpu().data.numpy()
    dataLoader_list[1].adata.obsm['ReX'] = trainer.adata2_output.cpu().data.numpy()

    dataLoader_list[0].adata.obsm['fused_embedding'] = trainer.adata1_fused.cpu().data.numpy()
    dataLoader_list[1].adata.obsm['fused_embedding'] = trainer.adata2_fused.cpu().data.numpy()

    return [dataLoader_list[0].adata, dataLoader_list[1].adata], trainer
