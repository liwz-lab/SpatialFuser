import torch
import torch.nn as nn
import torch.nn.functional as F
from .vis import matching_score


class OptimalMatchingLayer(nn.Module):
    def __init__(self, args):
        super(OptimalMatchingLayer, self).__init__()
        self.args = args
        self.bin_score = None
        self.roi_radius = self.args.roi_radius
        self.K = self.args.m_top_K

    def forward(self, adata1_embedding, adata2_embedding, adata1_loc, adata2_loc, iters=100):
        # Compute matching descriptor distance.
        roi_mask, self.scores = self.get_score_matrix(adata1_embedding, adata2_embedding, adata1_loc, adata2_loc)
        # self.bin_score = torch.tensor(1)
        self.bin_score = self.scores[self.scores != 0].median()
        soft_matches = self.log_optimal_transport(self.scores, self.bin_score, iters=iters)

        if self.K >= 1:
            match_in_adata1 = torch.tensor([], dtype=torch.int)
            match_in_adata2 = torch.tensor([], dtype=torch.int)

            all_mscore0 = torch.zeros(adata1_embedding.shape[0])[None].to(adata1_embedding)
            all_mscore1 = torch.zeros(adata2_embedding.shape[0])[None].to(adata2_embedding)

            topk0, topk1 = soft_matches[:, :-1, :-1].topk(self.K, dim=2), soft_matches[:, :-1, :-1].topk(self.K, dim=1)
            for k in range(self.K):
                indices0, indices1 = topk0.indices[:,:,k], topk1.indices[:,k,:]

                mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
                mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1)

                zero = soft_matches.new_tensor(0)
                # if MNN matching exist, return exp(reliability)
                mscores0 = torch.where(mutual0, topk0.values[:,:,k].exp(), zero)
                mscores1 = torch.where(mutual1, topk1.values[:,k,:].exp(), zero)

                all_mscore0 = all_mscore0 + mscores0
                all_mscore1 = all_mscore1 + mscores1

                # Get the matches with score above "match_threshold".

                # # quality control based on P-value
                # n_sampling = 1000
                # # null distribution sampling
                # match_score = soft_matches[:, :-1, :-1].exp()
                # none_zero_score = match_score[match_score > 1e-5]
                # sample_index = torch.randint(0, none_zero_score.shape[0], (n_sampling,))
                # null_distri = none_zero_score[sample_index]
                # null_distri = torch.sort(null_distri, descending=False)[0]
                # threshold = null_distri[int(0.9 * n_sampling)]
                # # filter
                # # valid0 = mutual0 & (mscores0 > threshold)

                match_threshold = 0
                valid0 = mutual0 & (mscores0 > match_threshold)
                valid1 = mutual1 & valid0.gather(1, indices1)

                indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
                indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

                temp_match_in_adata1 = torch.where(indices0 != -1)[1].data.cpu()
                temp_match_in_adata2 = indices0[0, temp_match_in_adata1].data.cpu()

                match_in_adata1 = torch.concat([match_in_adata1, temp_match_in_adata1])
                match_in_adata2 = torch.concat([match_in_adata2, temp_match_in_adata2])

            # vis score distribution
            if self.args.verbose:
                matching_score(adata1_embedding.data.cpu().numpy(),
                               adata1_loc.data.cpu().numpy(),
                               all_mscore0.data.cpu().numpy(),
                               1,
                               0.015)

                matching_score(adata2_embedding.data.cpu().numpy(),
                               adata2_loc.data.cpu().numpy(),
                               all_mscore1.data.cpu().numpy(),
                               2,
                               0.015)

            negative_in_adata1 = self.sample_negative_indices(adata1_loc, self.args.roi_radius).T[0,match_in_adata1]
            negative_in_adata2 = self.sample_negative_indices(adata2_loc, self.args.roi_radius).T[0,match_in_adata2]

            return match_in_adata1, match_in_adata2, negative_in_adata1, negative_in_adata2

    def get_score_matrix(self, adata1_embedding, adata2_embedding, adata1_loc, adata2_loc):
        # spatial distance
        loc_dis = torch.cdist(adata1_loc, adata2_loc, p = 2)

        roi_mask = loc_dis.clone()
        roi_mask[loc_dis < self.roi_radius] = 1
        roi_mask[loc_dis >= self.roi_radius] = 0

        # norm
        adata1_embedding_normed = F.normalize(adata1_embedding, p=2, dim=-1)
        adata2_embedding_normed = F.normalize(adata2_embedding, p=2, dim=-1)

        # cosine simi
        cross_modality_scores = torch.matmul(adata1_embedding_normed, adata2_embedding_normed.T)

        # L1 distance as cost matrix
        # cross_modality_scores = torch.cdist(adata1_embedding_normed, adata2_embedding_normed, p=1)

        # L2 norm
        # cross_modality_scores = torch.cdist(adata1_embedding_normed, adata2_embedding_normed, p=2)

        tau = self.args.tau
        scores = roi_mask * ((cross_modality_scores + 1)/(2 * tau + 1e-6))

        if scores.dim() == 2:
            scores = scores[None]
        return roi_mask, scores

    @staticmethod
    def arange_like(x, dim: int):
        return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor,
                                epsilon = 1, iters: int = 100) -> torch.Tensor:
        """
        Perform Sinkhorn Normalization in Log-space for stability
        :param Z: The cost matrix or similarity matrix to be normalized.
        :param log_mu: Latent dual variable (Lagrange multiplier) associated with the row marginal constraints.
        :param log_nu: Latent dual variable (Lagrange multiplier) associated with the column marginal constraints.
        :param epsilon: Entropic regularization parameter.
               When ε is large, the values of Z / ε become smaller, causing the output of `torch.logsumexp` to approximate the logarithm of the average.
               This results in a smoother transport plan, as a larger ε compresses the range of the cost matrix, reducing differences between alternative paths.
               When ε is small, the values of Z / ε become larger, and the output of `torch.logsumexp` approximates the maximum of the inputs.
               This makes the transport plan closer to the original optimal transport problem, since a smaller ε amplifies the differences in the cost matrix and thus preserves more detailed structural information.
        :param iters: The number of iterations to perform for the Sinkhorn normalization procedure.
        :return: The normalized transport plan in log-space.
        """
        actual_nits = 0
        thresh = 1e-4
        with torch.no_grad():
            u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
            for _ in range(iters):
                u1 = u
                u = log_mu - torch.logsumexp(Z / (epsilon + 1e-6) + v.unsqueeze(1), dim=2)
                v = log_nu - torch.logsumexp(Z / (epsilon + 1e-6) + u.unsqueeze(2), dim=1)
                # u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
                # v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
                err = (u - u1).abs().sum(-1).mean()

                actual_nits += 1
                if err.item() < thresh:
                    break
        if self.args.verbose:
            print('actual_nits:', actual_nits)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    def log_optimal_transport(self, scores: torch.Tensor, bin_score: torch.Tensor,
                              iters: int) -> torch.Tensor:
        """ Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)

        bins0 = bin_score.expand(b, m, 1).to(scores)
        bins1 = bin_score.expand(b, 1, n).to(scores)
        alpha = bin_score.expand(b, 1, 1).to(scores)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
        # expand to multi-head/multi-channel
        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, self.args.epsilon, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z

    def sample_negative_indices(self, loc, alpha=0.1):
        dist = torch.cdist(loc, loc, p=2)
        mask = dist > alpha
        mask_float = mask.float()
        indices = torch.multinomial(mask_float, num_samples=1)
        return indices
