import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, \
    v_measure_score


def show_para_num(model):
    """
    Count the trainable and non-trainable parameters of a model.

    Args:
        model: torch.nn.Module. The neural network model to be inspected.

    Returns:
        total_params: Total number of parameters in the model.
        total_trainable_params: Number of trainable parameters in the model.
    """

    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    return total_params, total_trainable_params


def mclust_R(adata, num_cluster, modelNames='EII', used_obsm='embedding', random_seed=2023):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    try:
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames, verbose=True)
        mclust_res = np.array(res[-2])
    except TypeError:
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, 'EII', verbose=True)
        mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata


def metrics(adata, save_loc, n_neighbors=10, resolution=0.1, spot_size=0.02, cluster_label='Region', mclust_model='EEE',
            plot_color: list = None, embed_label='embedding', vis=True, save=False):
    """
    Assess tissue domain detection tasks.
    This function automatically computes five clustering metrics (ARI, AMI, Homogeneity, Completeness, and V-Measure) under clustering methods including Leiden, Louvain, and Mclust.
    It also generates UMAP visualizations colored by ground truth labels and spatial distribution plots of the detected tissue domains.

    Args:
        adata: AnnData. The AnnData object to be visualized and evaluated.
        save_loc: str. Directory path to save the results.
        n_neighbors: int. The `sc.pp.neighbors` parameter controlling the number of neighbors prior to clustering.
        resolution: float. Resolution parameter for Leiden and Louvain clustering.
        spot_size: float. Spot size for spatial visualization.
        cluster_label: str. Column name in `adata.obs` containing the ground truth labels.
        mclust_model: str. Mclust clustering mode.
        plot_color: list or dict. Color scheme for plotting.
        embed_label: str. Key in `adata.obsm` specifying the embeddings used for clustering.
        vis: bool. Whether to visualize the results.
        save: bool. Whether to save the results to `save_loc`.

    Returns:
        leiden_result: Clustering metrics and results from Leiden clustering.
        louvain_result: Clustering metrics and results from Louvain clustering.
        mclust_result: Clustering metrics and results from Mclust clustering.
    """
    if plot_color is None:
        plot_color = ["leiden", "louvain", "mclust"]

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=embed_label)
    sc.tl.leiden(adata, resolution=resolution)
    sc.tl.louvain(adata, resolution=resolution)

    adata = mclust_R(adata, modelNames=mclust_model, used_obsm=embed_label,
                     num_cluster=adata.obs[cluster_label].dropna(axis=0).unique().shape[0])
    obs_df = adata.obs[cluster_label].dropna(axis=0)

    obs_df = adata.obs.dropna(axis=0)
    leiden_result = pd.DataFrame(index=['leiden'], columns=['ARI', 'AMI', 'Homogeneity', 'Completeness', 'V_Measure'])
    louvain_result = pd.DataFrame(index=['louvain'], columns=['ARI', 'AMI', 'Homogeneity', 'Completeness', 'V_Measure'])
    mclust_result = pd.DataFrame(index=['mclust'], columns=['ARI', 'AMI', 'Homogeneity', 'Completeness', 'V_Measure'])

    for clust_method in ['leiden', 'louvain', 'mclust']:
        locals()[clust_method + '_result'].loc[clust_method, 'ARI'] = adjusted_rand_score(obs_df[cluster_label],
                                                                                          obs_df[clust_method])
        locals()[clust_method + '_result'].loc[clust_method, 'AMI'] = adjusted_mutual_info_score(obs_df[cluster_label],
                                                                                                 obs_df[clust_method])
        locals()[clust_method + '_result'].loc[clust_method, 'Homogeneity'] = homogeneity_score(obs_df[cluster_label],
                                                                                                obs_df[clust_method])
        locals()[clust_method + '_result'].loc[clust_method, 'Completeness'] = completeness_score(obs_df[cluster_label],
                                                                                                  obs_df[clust_method])
        locals()[clust_method + '_result'].loc[clust_method, 'V_Measure'] = v_measure_score(obs_df[cluster_label],
                                                                                            obs_df[clust_method])

    if vis:
        sc.tl.umap(adata)

        titles = ['Original']
        method_titles = {
            'leiden': 'leiden ARI: {:.2f}'.format(leiden_result['ARI'][0]),
            'louvain': 'louvain ARI: {:.2f}'.format(louvain_result['ARI'][0]),
            'mclust': 'mclust ARI: {:.2f}'.format(mclust_result['ARI'][0]),
        }
        for method in plot_color:
            if method in method_titles:
                titles.append(method_titles[method])

        if save:
            sc.pl.umap(adata,
                       color=[cluster_label] + plot_color,
                       save=save_loc)

            sc.pl.spatial(adata, img_key="hires",
                          color=[cluster_label] + plot_color,
                          spot_size=spot_size,
                          title=titles,
                          wspace=0.5,
                          save=save_loc
                          )
        else:
            sc.pl.umap(adata,
                       color=[cluster_label] + plot_color)

            sc.pl.spatial(adata,
                          img_key="hires",
                          color=[cluster_label] + plot_color,
                          spot_size=spot_size,
                          title=titles,
                          wspace=0.5
                          )

    return leiden_result, louvain_result, mclust_result


def trajectory_analysis(adata, save_loc, cluster_label='Region', save=False):
    """
    Run PAGA trajectory analysis.

    Args:
        adata: AnnData. The annotated data matrix.
        save_loc: str. Path to save the output figures.
        cluster_label: str. Key in `adata.obs` used to color the UMAP plot.
        save: bool. Whether to save the results.

    Returns:
        None
    """
    used_adata = adata[~adata.obs[cluster_label].isin([None, np.nan])].copy()
    sc.tl.paga(used_adata, groups=cluster_label)
    # plt.rcParams["figure.figsize"] = (4, 3)
    if save:
        sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20, legend_loc='on data',
                           legend_fontoutline=2, show=True, save=save_loc)
    else:
        sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20, legend_loc='on data',
                           legend_fontoutline=2, show=True)


def all_matching(adata1, adata2, percentile, roi, save_loc=None, file_name=None):
    """
    Global matching module

    Args:
        adata1: anndata.AnnData. Reference AnnData object (typically the one with fewer spots).
        adata2: anndata.AnnData. Target AnnData object.
        percentile: float. Confidence threshold for probability-based quality control (e.g., 0.95).
        roi: float. Radius of local random sampling.
        save_loc: str. Directory path for saving result figures.
        file_name: str. Name of the result figures.

    Returns:
        valid_ratio: Proportion of valid matches after quality control.
        matching_accuracy: Accuracy of the matching between reference and target.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pySankey.sankey import sankey
    import torch
    import torch.nn.functional as F

    # ground sampling：random select negative pair within roi
    n_random_samples = 1000
    random_indices_adata1 = np.random.choice(adata1.n_obs, size=n_random_samples, replace=False)
    background_cos_sims = []

    for i in range(n_random_samples):
        idx1 = random_indices_adata1[i]
        spatial_point = adata1.obsm['spatial'][idx1]
        embed1 = adata1.obsm['fused_embedding'][idx1]

        spatial_dists = np.linalg.norm(adata2.obsm['spatial'] - spatial_point, axis=1)
        candidate_mask = spatial_dists < roi
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            continue

        idx2 = np.random.choice(candidate_indices)
        embed2 = adata2.obsm['fused_embedding'][idx2]

        dot_product = np.dot(embed1, embed2)
        norm_product = np.linalg.norm(embed1) * np.linalg.norm(embed2)
        cos_sim = dot_product / (norm_product + 1e-12)
        background_cos_sims.append(cos_sim)

    background_cos_sims = np.array(background_cos_sims)
    threshold = np.percentile(background_cos_sims, percentile)
    print(f"\nQC threshold：{percentile}%  -> {threshold:.4f}")

    mapping_results = np.full(adata1.n_obs, -1, dtype=int)

    for i in range(adata1.n_obs):
        current_spatial = adata1.obsm['spatial'][i]
        current_embed = adata1.obsm['fused_embedding'][i]
        spatial_dists = np.linalg.norm(adata2.obsm['spatial'] - current_spatial, axis=1)
        candidate_mask = spatial_dists < roi
        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            continue

        candidate_embeds = adata2.obsm['fused_embedding'][candidate_indices]
        dot_products = np.dot(candidate_embeds, current_embed)
        norm_product = np.linalg.norm(current_embed) * np.linalg.norm(candidate_embeds, axis=1)
        cosine_sims = dot_products / (norm_product + 1e-12)

        # QC
        valid_mask = cosine_sims >= threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        # chose the most similar one
        best_idx = valid_indices[np.argmax(cosine_sims[valid_indices])]
        mapping_results[i] = candidate_indices[best_idx]

    valid_count = np.sum(mapping_results != -1)
    valid_ratio = valid_count / adata1.n_obs
    print(f"valid matching ratio：{valid_ratio:.1%} ({valid_count}/{adata1.n_obs})")

    correct_matches = 0
    valid_indices = np.where(mapping_results != -1)[0]
    if len(valid_indices) > 0:
        for src_idx in valid_indices:
            tgt_idx = mapping_results[src_idx]
            src_region = str(adata1.obs['Region'].iloc[src_idx])
            tgt_region = str(adata2.obs['Region'].iloc[tgt_idx])
            if src_region == tgt_region:
                correct_matches += 1
        accuracy = correct_matches / len(valid_indices)
        print(f"spots mapping acc：{accuracy:.2%} ({correct_matches}/{len(valid_indices)})")
    else:
        print("no valid matching")

    # Sankey plot
    adata1_df = pd.DataFrame({
        'index': range(adata1.shape[0]),
        'x': adata1.obsm['spatial'][:, 0],
        'y': adata1.obsm['spatial'][:, 1],
        'celltype': adata1.obs['Region']
    })
    adata2_df = pd.DataFrame({
        'index': range(adata2.shape[0]),
        'x': adata2.obsm['spatial'][:, 0],
        'y': adata2.obsm['spatial'][:, 1],
        'celltype': adata2.obs['Region']
    })

    Region_matching = pd.DataFrame()
    Region_matching['adata1_Region'] = adata1_df['celltype'][valid_indices].values
    Region_matching['adata2_Region'] = adata2_df['celltype'][mapping_results[mapping_results != -1]].values

    sankey(Region_matching["adata1_Region"], Region_matching["adata2_Region"], aspect=20, fontsize=10)
    plt.subplots_adjust(left=0.3, right=0.65, top=0.95, bottom=0.05)
    plt.title(f"Matching Number: {valid_count}, Matching Accuracy: {accuracy:.2%}")
    if save_loc is not None:
        plt.savefig(save_loc+file_name+'_alignment.svg')
    plt.show()

    adata1_embedding = torch.tensor(adata1.obsm['fused_embedding'])
    adata2_embedding = torch.tensor(adata2.obsm['fused_embedding'])
    adata1_normed = F.normalize(adata1_embedding, dim=-1)
    adata2_normed = F.normalize(adata2_embedding, dim=-1)
    cos_sim_matrix = torch.matmul(adata1_normed, adata2_normed.T)
    cos_values = cos_sim_matrix.flatten().cpu().numpy()
    matched_cos_sims = []
    for i in valid_indices:
        idx2 = mapping_results[i]
        embed1 = adata1.obsm['fused_embedding'][i]
        embed2 = adata2.obsm['fused_embedding'][idx2]
        dot_product = np.dot(embed1, embed2)
        norm_product = np.linalg.norm(embed1) * np.linalg.norm(embed2)
        cos_sim = dot_product / (norm_product + 1e-12)
        matched_cos_sims.append(cos_sim)
    matched_cos_sims = np.array(matched_cos_sims)
    bins = np.linspace(min(cos_values.min(), matched_cos_sims.min()),
                       max(cos_values.max(), matched_cos_sims.max()), 100)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), height_ratios=[1, 1])
    ax_top.hist(matched_cos_sims, bins=bins, color='lightgreen', edgecolor='black')
    ax_top.set_ylabel('Matched', fontsize=10)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_position(('outward', 5))
    ax_top.tick_params(direction='out', length=3)
    n, _, _ = ax_bottom.hist(cos_values, bins=bins, color='skyblue', edgecolor='black')
    ax_bottom.invert_yaxis()
    ax_bottom.set_ylabel('All Pairs', fontsize=10)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)
    ax_bottom.spines['left'].set_position(('outward', 5))
    ax_bottom.tick_params(direction='out', length=3)
    ax_bottom.set_xlabel('Cosine Similarity')
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc+file_name+'_simi.svg')
    plt.show()

    if len(valid_indices) > 0:
        return valid_ratio, accuracy
    else:
        return 0, 0
