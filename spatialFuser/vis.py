import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Mapping, Optional, Union


def visualize_loss(loss_list, title='Loss Visualization'):
    """
    Display a line plot of the specified loss function.

    Args:
        loss_list: list. A list of loss values recorded during training.
        title: str. The title of the plot.
    """

    plt.plot(loss_list, label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def vis_global_att(trainer, spot_size=0.015, save=None):
    """
    When alpha is not 0, display the remote information propagation pathway established during training for the spot specified by args.check_global_att_index.

    Args:
        trainer: object. The training model instance.
        spot_size: float. The size of the spot.
        save: str or None. Path to save the output. If None, the result is not saved.
    """

    import matplotlib as mpl
    related_id, related_id_counts = np.unique(trainer.globle_att, return_counts = True)
    self_idx = np.where(related_id == trainer.args.check_global_att_index)[0]
    global_att = np.zeros(trainer.adata.shape[0])
    related_id_counts[self_idx] = 0

    a = 0.15
    arr_normalized = (related_id_counts - np.min(related_id_counts)) * (1 - a) / (np.max(related_id_counts) - np.min(related_id_counts)) + a
    arr_normalized[self_idx] = 1
    global_att[related_id.astype(int)] = arr_normalized
    trainer.adata.obs['global_att'] = global_att

    color_map = mpl.colormaps['RdPu']
    truncated_cmap = color_map.resampled(256)
    truncated_cmap = mpl.colors.ListedColormap(truncated_cmap(np.linspace(0.1, 0.9, 256)))

    sc.pl.spatial(trainer.adata,
                  img_key="hires",
                  color=['Region', 'global_att'],
                  color_map=truncated_cmap,
                  spot_size=spot_size,
                  save=save
                  )

    print(trainer.adata.obs.iloc[trainer.args.check_global_att_index]['Region'])


def checkBatch(adata1, adata2, save=None):
    """
    Display the UMAP plot showing batch effects between two slices.
    Cell type or region labels are stored in `obs['Region']`.
    By default, the integrated features are stored in `obsm['fused_embedding']`.

    Args:
        adata1: AnnData. The first slice data.
        adata2: AnnData. The second slice data.
        save: str or None. Path to save the figure. If None, the plot will not be saved.
    """

    adata = sc.AnnData(X=np.concatenate([adata1.obsm['fused_embedding'], adata2.obsm['fused_embedding']]))
    adata.obsm['spatial'] = np.concatenate([adata1.obsm['spatial'], adata2.obsm['spatial']])
    adata.obs = pd.concat([adata1.obs, adata2.obs])
    adata.obs['batches'] = adata.obs['batches'].astype('category')
    sc.pp.neighbors(adata, n_neighbors = 15)
    sc.tl.umap(adata, min_dist=0.5, spread=1)
    sc.pl.umap(adata, color=["Region", "batches"], wspace=0.5, save=save)


class match_3D_multi():
    """
    Visualize the results of pair-wise slice alignment.

    Functions:
        draw_3D: Draw a 3D visualization of two datasets.
        draw_lines: Plot lines connecting paired cells between two datasets.
        draw_lines_sub_type: Plot lines connecting paired cells of specific cell types between two datasets.
    """

    def __init__(self, dataset_A: pd.DataFrame,
                 dataset_B: pd.DataFrame,
                 matching: np.ndarray,
                 meta: Optional[str] = None,
                 expr: Optional[str] = None,
                 subsample_size: Optional[int] = 300,
                 reliability: Optional[np.ndarray] = None,
                 scale_coordinate: Optional[bool] = True,
                 rotate: Optional[List[str]] = None,
                 exchange_xy: Optional[bool] = False,
                 subset: Optional[List[int]] = None
                 ) -> None:
        self.dataset_A = dataset_A.copy()
        self.dataset_B = dataset_B.copy()
        self.meta = meta
        self.matching = matching
        self.conf = reliability
        self.subset = subset  # index of query cells to be plotted
        scale_coordinate = True if rotate != None else scale_coordinate

        assert all(item in dataset_A.columns.values for item in ['index', 'x', 'y'])
        assert all(item in dataset_B.columns.values for item in ['index', 'x', 'y'])

        if meta:
            set1 = list(set(self.dataset_A[meta]))
            set2 = list(set(self.dataset_B[meta]))
            self.celltypes = set1 + [x for x in set2 if x not in set1]
            self.celltypes.sort()  # make sure celltypes are in the same order
            overlap = [x for x in set2 if x in set1]
            print(f"dataset1: {len(set1)} cell types; dataset2: {len(set2)} cell types; \n\
                    Total :{len(self.celltypes)} celltypes; Overlap: {len(overlap)} cell types \n\
                    Not overlap :[{[y for y in (set1 + set2) if y not in overlap]}]"
                  )
        self.expr = expr if expr else False

        if scale_coordinate:
            for i, dataset in enumerate([self.dataset_A, self.dataset_B]):
                for axis in ['x', 'y']:
                    dataset[axis] = (dataset[axis] - np.min(dataset[axis])) / (
                                np.max(dataset[axis]) - np.min(dataset[axis]))
                    if rotate == None:
                        pass
                    elif axis in rotate[i]:
                        dataset[axis] = 1 - dataset[axis]
        if exchange_xy:
            self.dataset_B[['x', 'y']] = self.dataset_B[['y', 'x']]

        if not subset is None:
            matching = matching[:, subset]
        if matching.shape[1] > subsample_size and subsample_size > 0:
            self.matching = matching[:, np.random.choice(matching.shape[1], subsample_size, replace=False)]
        else:
            subsample_size = matching.shape[1]
            self.matching = matching
        print(f'Subsampled {subsample_size} pairs from {matching.shape[1]}')

        self.datasets = [self.dataset_A, self.dataset_B]

    def draw_3D(self,
                target: Union['all_type', List[str]] = 'all_type',
                size: Optional[List[int]] = [10, 10],
                conf_cutoff: Optional[float] = 0,
                point_size: Optional[List[int]] = [0.1, 0.1],
                line_width: Optional[float] = 0.3,
                line_color: Optional[str] = 'grey',
                line_alpha: Optional[float] = 0.7,
                hide_axis: Optional[bool] = False,
                show_error: Optional[bool] = False,
                show_celltype: Optional[bool] = False,
                only_show_correct: Optional[bool] = False,
                only_show_error: Optional[bool] = False,
                cmap: Optional[bool] = 'Reds',
                save: Optional[str] = None
                ) -> None:
        r"""
        Draw 3D picture of two datasets

        Parameters:
        ----------
        size
            plt figure size
        conf_cutoff
            confidence cutoff of mapping to be plotted
        point_size
            point size of every dataset
        line_width
            pair line width
        line_color
            pair line color
        line_alpha
            pair line alpha
        hide_axis
            if hide axis
        show_error
            if show error celltype mapping with different color
        cmap
            color map when vis expr
        save
            save file path
        """
        self.conf_cutoff = conf_cutoff
        show_error = show_error if self.meta else False
        fig = plt.figure(figsize=(size[0], size[1]))
        ax = fig.add_subplot(111, projection='3d')
        # color by meta
        if self.meta:
            # color = get_color(len(self.celltypes))
            cmap = plt.get_cmap('tab20')
            color = [cmap(i / len(self.celltypes)) for i in range(len(self.celltypes))]
            c_map = {celltype: color[i] for i, celltype in enumerate(self.celltypes)}

            if self.expr:
                # 在表达数据存在的情况下，直接使用 cmap 和 norm
                for i, dataset in enumerate(self.datasets):
                    norm = plt.Normalize(dataset[self.expr].min(), dataset[self.expr].max())
                    for cell_type in self.celltypes:
                        slice = dataset[dataset[self.meta] == cell_type]
                        xs = slice['x']
                        ys = slice['y']
                        zs = i
                        ax.scatter(xs, ys, zs, s=point_size[i], c=slice[self.expr], cmap=cmap, norm=norm)
            else:
                # 在表达数据不存在的情况下，使用 c_map
                for i, dataset in enumerate(self.datasets):
                    for cell_type in self.celltypes:
                        slice = dataset[dataset[self.meta] == cell_type]
                        xs = slice['x']
                        ys = slice['y']
                        zs = i
                        ax.scatter(xs, ys, zs, s=point_size[i], c=[c_map[cell_type]] * len(xs))
        # plot points without meta
        else:
            for i, dataset in enumerate(self.datasets):
                xs = dataset['x']
                ys = dataset['y']
                zs = i
                ax.scatter(xs, ys, zs, s=point_size[i])
        # plot line
        self.c_map = c_map
        if target == 'all_type':
            self.draw_lines(ax, show_error, show_celltype, only_show_correct, only_show_error, line_color, line_width, line_alpha)
        else:
            self.draw_lines_sub_type(target, ax, show_error, only_show_correct, only_show_error, line_color, line_width, line_alpha)
        if hide_axis:
            plt.axis('off')
        if save != None:
            plt.savefig(save)
        plt.show()

    def draw_lines(self, ax, show_error, show_celltype, only_show_correct, only_show_error, line_color, line_width=0.3, line_alpha=0.7) -> None:
        r"""
        Draw lines between paired cells in two datasets
        """
        for i in range(self.matching.shape[1]):
            if not self.conf is None and self.conf[i] < self.conf_cutoff:
                continue
            pair = self.matching[:, i]
            default_color = line_color
            draw_line = True

            if self.meta != None:
                try:
                    celltype1 = self.dataset_A.loc[self.dataset_A['index'] == pair[0], self.meta].astype(str).values[0]
                    celltype2 = self.dataset_B.loc[self.dataset_B['index'] == pair[1], self.meta].astype(str).values[0]
                except IndexError:
                    print('index: ', i)
                if show_error:
                    if celltype1 == celltype2:
                        color = '#ade8f4'  # blue
                    else:
                        color = '#ffafcc'  # red
                if show_celltype:
                    if celltype1 == celltype2:
                        color = self.c_map[celltype1]
                    else:
                        color = '#696969'  # celltype1 error match color
                if only_show_correct:
                    if celltype1 == celltype2:
                        color = '#ade8f4'  # blue
                    else:
                        draw_line = False
                if only_show_error:
                    if celltype1 != celltype2:
                        color = '#ffafcc'  # red
                    else:
                        draw_line = False

            if draw_line:
                point0 = np.append(self.dataset_A[self.dataset_A['index'] == pair[0]][['x', 'y']], 0)
                point1 = np.append(self.dataset_B[self.dataset_B['index'] == pair[1]][['x', 'y']], 1)

                coord = np.row_stack((point0, point1))
                color = color if show_error or show_celltype or only_show_correct or only_show_error else default_color
                ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], color=color, linestyle="dashed", linewidth=line_width,
                        alpha=line_alpha)

    def draw_lines_sub_type(self, target, ax, show_error, only_show_correct, only_show_error, line_color, line_width=0.3, line_alpha=0.7) -> None:
        r"""
        Draw lines between paired cells in two datasets
        """
        for i in range(self.matching.shape[1]):

            if not self.conf is None and self.conf[i] < self.conf_cutoff:
                continue
            pair = self.matching[:, i]
            default_color = line_color
            draw_line = True

            if self.meta != None:
                try:
                    celltype1 = self.dataset_A.loc[self.dataset_A['index'] == pair[0], self.meta].astype(str).values[0]
                    celltype2 = self.dataset_B.loc[self.dataset_B['index'] == pair[1], self.meta].astype(str).values[0]
                except IndexError:
                    print('index: ', i)

                if show_error:
                    if celltype1 == celltype2:
                        color = '#ade8f4'  # blue
                    else:
                        color = '#ffafcc'  # red

                if only_show_correct:
                    if celltype1 == celltype2:
                        color = '#ade8f4'  # blue
                    else:
                        draw_line = False
                if only_show_error:
                    if celltype1 != celltype2:
                        color = '#ffafcc'  # red
                    else:
                        draw_line = False

            if celltype2 not in target:
                draw_line = False

            if draw_line:
                point0 = np.append(self.dataset_A[self.dataset_A['index'] == pair[0]][['x', 'y']], 0)
                point1 = np.append(self.dataset_B[self.dataset_B['index'] == pair[1]][['x', 'y']], 1)

                coord = np.row_stack((point0, point1))
                color = color if show_error or only_show_correct or only_show_error else default_color
                ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], color=color, linestyle="dashed", linewidth=line_width,
                        alpha=line_alpha)


def matching_score(emb, loc, score, slice_id, spot_size):
    adata = sc.AnnData(X=emb)
    adata.obsm['spatial'] = loc
    adata.obs['score'] = score.T
    sc.pl.spatial(adata,
                  color='score',
                  spot_size=spot_size,
                  title='matching soft cost value in slice {}'.format(slice_id)
                  )
