from pymnet import MultilayerNetwork
from pymnet import draw as pymnet_draw
def vizualize_layers(A,show=True,layerOrderDict={},figsize=(42, 25),nodeSizeRule={"rule": "degree", "propscale": 0.0005}):
    '''

    Args:
        A: multiplex
        show: show plot
        layerOrderDict: the order of the layers for vizualization
        figsize: the size of the figure
        nodeSizeRule: the rule for the node size e.g. degree

    Returns:

    '''

    mplex = MultilayerNetwork(aspects=1);
    for g in A.get_edges(True):
             mplex[g[0][0], g[1][0], g[0][1], g[1][1]] = 1

    pymnet_draw(mplex, show=show,
                                      figsize=figsize, layerPadding=0.6, layergap=0.6, defaultLayerAlpha=0.3,
                                      layout="spring", elev=8, azim=5,
                                      nodeColorDict={(0, 0): "r", (1, 0): "r", (0, 1): "r"},
                                      nodeLabelRule={}, defaultLayerLabelLoc=(0, 1),
                                      layerOrderDict=layerOrderDict,
                                      defaultLayerLabelSize=18,
                                      edgeColorRule={"rule": "edgeweight", "colormap": "jet", "scaleby": 0.1},
                                      nodeSizeRule=nodeSizeRule);