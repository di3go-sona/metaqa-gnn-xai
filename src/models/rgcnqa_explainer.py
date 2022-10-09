#%%
EPS = 1e-15

from torch_geometric.nn import GNNExplainer
from typing import Optional

from math import sqrt
from inspect import signature
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx


class RGCNQAExplainer(GNNExplainer):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        num_hops (int, optional): The number of hops the :obj:`model` is
            aggregating information from.
            If set to :obj:`None`, will automatically try to detect this
            information based on the number of
            :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`
            layers inside :obj:`model`. (default: :obj:`None`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
            returns the logarithm of probabilities), :obj:`"prob"` (the
            model returns probabilities), :obj:`"raw"` (the model returns raw
            scores) and :obj:`"regression"` (the model returns scalars).
            (default: :obj:`"log_prob"`)
        feat_mask_type (str, optional): Denotes the type of feature mask
            that will be learned. Valid inputs are :obj:`"feature"` (a single
            feature-level mask for all nodes), :obj:`"individual_feature"`
            (individual feature-level masks for each node), and :obj:`"scalar"`
            (scalar mask for each each node). (default: :obj:`"feature"`)
        allow_edge_mask (boolean, optional): If set to :obj:`False`, the edge
            mask will not be optimized. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
    """

    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, **kwargs):
        super().__init__(model,
                        epochs, 
                        lr, 
                        num_hops, 
                        return_type, 
                        feat_mask_type, 
                        allow_edge_mask, 
                        log, **kwargs)

   
    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, edge_y=None, node_alpha=None,
                           seed=10, **kwargs):
        r"""Visualizes the subgraph given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`-1` to explain graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node.
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None

        else:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                
                [node_idx, kwargs['src_idx']], self.num_hops, edge_index, 
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        if edge_y is None:
            edge_color = ['black'] * edge_index.size(1)
        else:
            colors = list(plt.rcParams['axes.prop_cycle'])
            edge_color = [
                colors[i % len(colors)]['color']
                for i in edge_y[hard_edge_mask]
            ]

        data = Data(edge_index=edge_index, att=edge_mask,
                    edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'],
                        edge_attrs=['att', 'edge_color'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
        node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
        label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G, seed=seed)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    color=data['edge_color'],
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))

        if node_alpha is None:
            nx.draw_networkx_nodes(G, pos, node_color=y.tolist(),
                                   **node_kwargs)
        else:
            node_alpha_subset = node_alpha[subset]
            assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
            nx.draw_networkx_nodes(G, pos, alpha=node_alpha_subset.tolist(),
                                   node_color=y.tolist(), **node_kwargs)

        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
            self.__subgraph__(node_idx, x, edge_index, **kwargs)
        kwargs['src_idx'] = mapping.item()
        # Get the initial prediction.
        with torch.no_grad():

            
            out = self.model(x=x, edge_index=edge_index, **kwargs)
            if self.return_type == 'regression':
                prediction = out
            else:
                log_logits = self.__to_log_prob__(out)
                pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        if self.allow_edge_mask:
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()

            out = self.model(x=h, edge_index=edge_index, **kwargs)
            if self.return_type == 'regression':
                loss = self.__loss__(mapping, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(mapping, log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        if self.feat_mask_type == 'individual_feature':
            new_mask = x.new_zeros(num_nodes, x.size(-1))
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        elif self.feat_mask_type == 'scalar':
            new_mask = x.new_zeros(num_nodes, 1)
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        node_feat_mask = node_feat_mask.squeeze()

        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask
    def __repr__(self):
        return f'{self.__class__.__name__}()'

# %%
