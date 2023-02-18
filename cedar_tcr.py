import os.path

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from finetuning_scheduler import FinetuningScheduler
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
import pandas as pd
import csv
import itertools
import pytorch_lightning as pl
import torchmetrics
from Bio import SeqIO
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from py3plex.algorithms.statistics.basic_statistics import identify_n_hubs, core_network_statistics
from py3plex.core import converters
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split, DataLoader
from torchmetrics import F1Score, AUROC
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall

from models import HeteroGNN
from numpy import linalg as LA
from pymnet import draw as pymnet_draw
from pymnet import MultilayerNetwork, MultiplexNetwork
import math
import networkx as nx
from py3plex.core import multinet
import torch
import numpy as np


#
# def prepare_training_data(patient, hla_alleles,output_file,tcell_table_file):
#     # read tcell assays table
#     assay_dict = {}
#     with open(tcell_table_file, 'r',encoding='utf-8') as input_handle:
#         csv_reader = csv.reader(input_handle, delimiter=',')
#         header_1 = next(csv_reader)
#         header_2 = next(csv_reader)
#         header_list = []
#         for x, y in zip(header_1, header_2):
#             header_list.append(':'.join([x, y]))
#         for row in csv_reader:
#             assert len(row) == len(header_list)
#             assay = {}
#             for x, y in zip(header_list, row):
#                 assay[x] = y
#             assay_id = assay['Reference:Assay IRI']
#             assay_dict[assay_id] = assay
#     print("len(assay_dict) =", len(assay_dict))
#
#     print("patient =", patient)
#     print("hla_alleles =", hla_alleles)
#     # assay_filtered = []
#     # for assay in assay_dict.values():
#     #     allele = assay['MHC:Allele Name']
#     #     epitope_type = assay['Epitope:Object Type']
#     #     if hla_alleles:
#     #         if (allele in hla_alleles and epitope_type == "Linear peptide"):
#     #             assay_filtered.append(assay)
#     #     else:
#     #         if (epitope_type == "Linear peptide"):
#     #             assay_filtered.append(assay)
#
#     print("len(assay_filtered) =", len(assay_dict.values()))
#     peptide_set = set([x['Epitope:Description'] for x in assay_dict.values()])
#     print("len(peptide_set) =", len(peptide_set))
#     print()
#
#     # output temporary csv
#     with open(output_file, 'w') as output_handle:
#         # fieldnames = ['Reference:T Cell ID',
#         #               'Reference:Reference ID',
#         #               'MHC:Allele Name',
#         #               'Epitope:Epitope ID',
#         #               'Epitope:Object Type',
#         #               'Epitope:Description',
#         #               'Assay:Qualitative Measure',
#         #               ]
#         fieldnames = assay_dict.values()[0].keys()
#         csv_writer = csv.DictWriter(output_handle, fieldnames, delimiter=',')
#         csv_writer.writeheader()
#         for assay in assay_dict.values():
#             assay_subset = {k: v for k, v in assay.items() if k in fieldnames}
#             csv_writer.writerow(assay_subset)
# aa_list = ['_PAD',
#            'A',
#            'R',
#            'N',
#            'D',
#            'C',
#            'E',
#            'Q',
#            'G',
#            'H',
#            'I',
#            'L',
#            'K',
#            'M',
#            'F',
#            'P',
#            'S',
#            'T',
#            'W',
#            'Y',
#            'V',
#           ]
# vocab_size = len(aa_list)
# aa2index = {}
# index2aa = {}
# for index, aa in enumerate(aa_list):
#     aa2index[aa] = index
#     index2aa[index] = aa
def vizualize_layers(A, show=True, layerOrderDict={}, figsize=(42, 25),
                     nodeSizeRule={"rule": "degree", "propscale": 0.05}, elev=8, azim=5,
                     edgeColorRule={"rule": "edgeweight", "colormap": "jet", "scaleby": 0.5}):
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
                layout="spring", elev=elev, azim=azim,
                nodeColorDict={(0, 0): "r", (1, 0): "r", (0, 1): "r"},
                nodeLabelRule={}, defaultLayerLabelLoc=(0, 1),
                layerOrderDict=layerOrderDict,
                defaultLayerLabelSize=18,
                edgeColorRule=edgeColorRule,
                nodeSizeRule=nodeSizeRule);


def viz_newtwork(A, name=None):
    mplex = MultilayerNetwork(aspects=1)
    for g in A.get_edges(True):
        mplex[g[0][0], g[1][0], g[0][1], g[1][1]] = 1
    fig = pymnet_draw(mplex, show=True,
                      figsize=(25, 25), layerPadding=0.6, layergap=0.6, defaultLayerAlpha=0.3, layout="spring", elev=18,
                      azim=55, defaultLayerLabelLoc=(1, 1),
                      nodeColorDict={(0, 0): "r", (1, 0): "r", (0, 1): "r"},
                      nodeLabelRule={},
                      defaultLayerLabelSize=12,
                      edgeColorRule={"rule": "edgeweight", "colormap": "jet", "scaleby": 0.1},
                      nodeSizeRule={"rule": "degree", "propscale": 0.05})
    if name:
        fig.savefig(name)


def compute_attributes(data, layers=['system', 'complex', 'polymer', 'monomer', 'atom', 'coord']):
    pca = TruncatedSVD(1)
    _df = []
    if not isinstance(data, list):
        data = [data]
    for system in data:

        matched_pbmc = list(map(lambda x: x[-1]['y'],
                                list(filter(lambda x: 'system' in x[0], list(system.core_network.nodes(True))))))[0]
        system = system.subnetwork(layers, subset_by="layers")
        # vectors=np.cov(np.squeeze(np.array(list(map(lambda x: x[-1] ,system.core_network.nodes(data='x')   ))))).mean()
        # vectors=np.squeeze(np.array(list(map(lambda x: x[-1] ,system.core_network.nodes(data='x')   )))).mean()
        # -----------------------Multiplex measures-----------------------
        # -------------- Overlapping degree--------------
        layer_names, separate_layers, multiedges = converters.prepare_for_parsing(system.core_network)
        overlapping_degree = dict()
        for node_highest_layer in separate_layers[layer_names.index('complex')].nodes():
            for layer_name, layer in zip(layer_names, separate_layers):
                overlapping_degree[(node_highest_layer, layer_name)] = np.sum([layer.degree[n] for n in
                                                                               nx.single_source_shortest_path_length(
                                                                                   system.core_network,
                                                                                   node_highest_layer) if
                                                                               n[-1] == layer_name])
        average_overlapping_degree = np.sum(list(overlapping_degree.values())) / (list(system.get_nodes())).__len__()

        # --------------Multiplex Participation Coefficient --------------
        M = layer_names.__len__()
        P = []
        for node_highest_layer in separate_layers[layer_names.index('complex')].nodes():
            o_i = 0
            for layer_name, layer in zip(layer_names, separate_layers):
                o_i += np.sum([layer.degree[n] for n in
                               nx.single_source_shortest_path_length(system.core_network, node_highest_layer) if
                               n[-1] == layer_name])
            for layer_name, layer in zip(layer_names, separate_layers):
                d_i = np.sum([layer.degree[n] for n in
                              nx.single_source_shortest_path_length(system.core_network, node_highest_layer) if
                              n[-1] == layer_name])
                p = (M / (M - 1)) * 1 - (d_i / o_i) ** 2
                P.append(p)
        average_participation_coefficient = np.sum(P) / (list(system.get_nodes())).__len__()
        layer.adjacency()
        # --------------Multiplex node   interdependence --------------
        # complexes=list(combinations(
        #     [node_highest_layer for node_highest_layer in separate_layers[layer_names.index('complex')].nodes()], 2))
        # for node_highest_layer in separate_layers[layer_names.index('complex')].nodes():
        #     path_length=nx.single_source_shortest_path_length(system.core_network, node_highest_layer)
        #     for layer_name, layer in zip(layer_names, separate_layers):
        # Supra centrality
        layers_pagerank_centr = 0
        for layer_name, layer in zip(layer_names, separate_layers):
            layers_pagerank_centr += sum(nx.pagerank(layer).values()) / len(nx.nodes(layer))

        avg_layers_pagerank_centr = layers_pagerank_centr / layer_names.__len__()
        # ----------------------- End Multiplex measures -----------------------
        vectors = np.squeeze(pca.fit_transform(
            csr_matrix(list(map(lambda x: np.squeeze(x[-1]), system.core_network.nodes(data='x')))).reshape(1, -1)))
        G = nx.Graph(system.core_network)

        G.remove_edges_from(nx.selfloop_edges(G))
        G = G.to_undirected()

        normalizedL = nx.normalized_laplacian_matrix(G).todense();
        eigen_values, eigen_vectors = LA.eigh(normalizedL);

        v_2 = np.squeeze(np.asarray(eigen_vectors[:, -2]));
        node_color = np.ones((v_2.shape[0], 1)).astype(float)
        # # Assign 0.0 for the blue color
        node_color[v_2 < 0] = 0.0;
        #         hubs, autho = nx.hits(G)

        #         h_a = np.array(np.array(list(hubs.values())) > np.array(list(autho.values()))).astype(int).mean()

        trans = nx.transitivity(G);
        nr_nodes = system.monoplex_nx_wrapper("number_of_nodes")
        nr_edges = system.monoplex_nx_wrapper("number_of_edges");
        largest_cc = max(nx.connected_components(G), key=len)

        s_1 = len(largest_cc) / np.float32(nr_nodes)
        degree_centrality = (1.0 / nr_nodes) * sum(system.monoplex_nx_wrapper("degree_centrality").values())
        average_betweenness_centrality = (1.0 / nr_nodes) * sum(nx.betweenness_centrality(G).values())
        closeness_centrality = (1.0 / nr_nodes) * sum(nx.closeness_centrality(G).values())
        try:
            current_flow_closeness_centrality = sum(nx.current_flow_closeness_centrality(G).values()) / len(nx.nodes(G))
        except Exception as exc:
            current_flow_closeness_centrality = 0
        degree_pearson_correlation_coefficient = nx.degree_pearson_correlation_coefficient(G)
        attribute_assortativity_coefficient_type = nx.attribute_assortativity_coefficient(G, 'type')
        attribute_assortativity_coefficient_source = nx.attribute_assortativity_coefficient(G, 'source')

        try:
            eigenvector_centrality = sum(nx.eigenvector_centrality(G).values()) / len(nx.nodes(G))
        except Exception as exc:
            eigenvector_centrality = 0
        pagerank_centrality = sum(nx.pagerank(G).values()) / len(nx.nodes(G))
        # Watts and Strogatz clustering coefficient
        clustering_sequence = [d for d in nx.clustering(G).values()]
        global_clus = (1.0 / nr_nodes) * sum(clustering_sequence)
        avg_core_hubs = (1.0 / nr_nodes) * sum(list(identify_n_hubs(system.core_network).values()))

        try:
            density = nx.density(G)
        except Exception as exc:
            density = 0
        try:
            diameter = nx.diameter(system.core_network)
        except Exception as exc:
            diameter = 0

        h, a = nx.hits(system.core_network)
        avg_v_hubs = (1.0 / nr_nodes) * np.sum([v for v in h.values()])
        avg_v_authorities = (1.0 / nr_nodes) * np.sum([v for v in a.values()])
        d = {
            'avg_layers_pagerank_centr': avg_layers_pagerank_centr,
            'average_participation_coefficient': average_participation_coefficient,
            'average_overlapping_degree': average_overlapping_degree,
            's_largest_component': s_1,
            'laplacianv_2': node_color.mean(),
            #            'hubs_auth': h_a,
            'global_clus_coefficient': global_clus,
            'transitivity': trans,
            'average_k': 2 * nr_edges / nr_nodes,
            'average_clustering': nx.average_clustering(G),
            'degree_centrality': degree_centrality,
            'avg_core_hubs': avg_core_hubs,
            "diameter": diameter,
            'density': density,
            'vectors': vectors,
            #            'avg_v_hubs':avg_v_hubs,
            #            'avg_v_authorities':avg_v_authorities,
            'average_betweenness_centrality': average_betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'current_flow_closeness_centrality': current_flow_closeness_centrality,
            'degree_pearson_correlation_coefficient': degree_pearson_correlation_coefficient,
            'eigenvector_centrality': eigenvector_centrality,
            'pagerank_centrality': pagerank_centrality,
            'attribute_assortativity_coefficient_type': attribute_assortativity_coefficient_type,
            'attribute_assortativity_coefficient_source': attribute_assortativity_coefficient_source,
            'matched_pbmc': matched_pbmc}

        for k in d.keys():
            if math.isnan(d[k]):
                d[k] = 0

        _df.append(d)
    return _df


IEDB_response_code = {'Positive': 1,
                      'Positive-High': 1,
                      'Positive-Intermediate': 1,
                      'Positive-Low': 1,
                      'Negative': 0,
                      }
MAX_LEN = 15


# allellist=pd.read_csv(allellist_file, sep=",")
# allellist=allellist.astype({column:'string' for column in allellist.columns})

# fasta_sequences = SeqIO.parse(open(hlalleles_prot_fastas_seq),'fasta')
# alles_fastea_seq_df=pd.DataFrame([ {'AlleleID':str(fasta.id),"Description": fasta.description,"Sequence":str(fasta.seq)} for fasta in fasta_sequences])
# alles_fastea_seq_df=alles_fastea_seq_df.astype({column:'string' for column in alles_fastea_seq_df.columns})
# alles_fastea_seq_df['AlleleID']=alles_fastea_seq_df['AlleleID'].apply(lambda x:x.replace('HLA:',''))
#
# allellist=allellist.merge(alles_fastea_seq_df,on='AlleleID')
# allellist=allellist.drop_duplicates()
# allellist['Sequence']=allellist['Sequence'].str.strip()
#
#
# with open(tcell_table_file, 'r', encoding='utf-8') as input_handle:
#     csv_reader = csv.reader(input_handle, delimiter=',')
#     header_1 = next(csv_reader)
#     header_2 = next(csv_reader)
#     header_list = []
#     for x, y in zip(header_1, header_2):
#         header_list.append(':'.join([x, y]))
# # patient = "mel15"
# # hla_alleles = ["HLA-A*03:01", "HLA-A*68:01", "HLA-B*27:05", "HLA-B*35:03", "HLA-C*02:02", "HLA-C*04:01"]
# # prepare_training_data(patient, None,output_file,tcell_table_file)
# mhc_full=pd.read_csv(tcell_table_file,skiprows=2)
# mhc_full.columns=header_list
# mhc_full['Allele']=mhc_full['MHC:Allele Name'].apply(lambda x: x.replace("HLA-",''))
# mhc_full['Epitope:Description']=mhc_full['Epitope:Description'].str.strip()
# idx=mhc_full['Epitope:Description'].str.contains(' ')
# mhc_full.loc[idx,'Epitope:Description']=mhc_full.loc[idx,'Epitope:Description'].apply(lambda x: x[0:x.index(' ')])
# mhc_filtered=mhc_full[(mhc_full['Allele'].isin(allellist.Allele))]
#
# mhc_filtered=mhc_filtered[mhc_filtered['Epitope:Parent Species']=='Homo sapiens']
#
# mhc_filtered=mhc_filtered.merge(allellist,on='Allele')
# mhc_filtered.rename(columns={"Description": "Allele:Description", "Sequence": "Allele:Sequence"})
#
#
# mhc_filtered['y']=mhc_filtered['Assay:Qualitative Measure'].apply(lambda x: IEDB_response_code[x])
#

def calculate_measures(input_df):
    device = torch.device(torch.cuda.current_device())
    heter_model = HeteroGNN(hidden=32, layers=16, nr_classes=6).to(device)
    result = []
    for index, e in input_df.iterrows():
        A = multinet.multi_layer_network(network_type="multiplex", directed=True)
        nodes = [{'source': 'system_{}_{}'.format(e['Epitope:Description'], e['Sequence']), 'type': 'system',
                  'x': np.random.random_sample(heter_model.hidden), 'y': e['y']},
                 {'source': 'complex_{}_{}'.format(e['Epitope:Description'], e['Sequence']), 'type': 'complex',
                  'x': np.random.random_sample(heter_model.hidden)},
                 {'source': 'polymer_{}'.format(e['Sequence']), 'type': 'polymer',
                  'x': np.random.random_sample(heter_model.hidden)},
                 {'source': 'polymer_{}'.format(e['Epitope:Description']), 'type': 'polymer',
                  'x': np.random.random_sample(heter_model.hidden)}] + \
                [{'source': '{}_{}'.format(e['Epitope:Description'], amino), 'type': 'monomer',
                  'x': heter_model.get_emb_amino_one_letter(
                      torch.LongTensor([heter_model.aminoAcids.index(amino)]).to(device)).detach().cpu().numpy()} for
                 amino in e['Epitope:Description']] \
                + [{'source': '{}_{}'.format(e['Sequence'], amino), 'type': 'monomer',
                    'x': heter_model.get_emb_amino_one_letter(
                        torch.LongTensor([heter_model.aminoAcids.index(amino)]).to(device)).detach().cpu().numpy()} for
                   amino in e['Sequence']]
        edges = [{'source': 'system_{}_{}'.format(e['Epitope:Description'], e['Sequence']),
                  'target': 'complex_{}_{}'.format(e['Epitope:Description'], e['Sequence']), 'type': 'has',
                  'source_type': 'system', 'target_type': 'complex'},
                 {'source': 'complex_{}_{}'.format(e['Epitope:Description'], e['Sequence']),
                  'target': 'polymer_{}'.format(e['Sequence']), 'type': 'has', 'source_type': 'complex',
                  'target_type': 'polymer'},
                 {'source': 'complex_{}_{}'.format(e['Epitope:Description'], e['Sequence']),
                  'target': 'polymer_{}'.format(e['Epitope:Description']), 'type': 'has', 'source_type': 'complex',
                  'target_type': 'polymer'}] \
                + [{'source': 'polymer_{}'.format(e['Sequence']), 'target': '{}_{}'.format(e['Sequence'], amino),
                    'type': 'has', 'source_type': 'polymer', 'target_type': 'monomer'} for amino in e['Sequence']] \
                + [{'source': 'polymer_{}'.format(e['Epitope:Description']),
                    'target': '{}_{}'.format(e['Epitope:Description'], amino), 'type': 'has', 'source_type': 'polymer',
                    'target_type': 'monomer'} for amino in e['Epitope:Description']]
        for i, amino in enumerate(e['Epitope:Description']):

            prev_amino = amino
            if i > 0:
                edges.append({'source': '{}_{}'.format(e['Epitope:Description'], prev_amino),
                              'target': '{}_{}'.format(e['Epitope:Description'], amino), 'type': 'chains_to',
                              'source_type': 'monomer',
                              'target_type': 'monomer'})

        for i, amino in enumerate(e['Sequence']):

            prev_amino = amino
            if i > 0:
                edges.append({'source': '{}_{}'.format(e['Sequence'], prev_amino),
                              'target': '{}_{}'.format(e['Sequence'], amino), 'type': 'chains_to',
                              'source_type': 'monomer',
                              'target_type': 'monomer'})
        A.add_nodes(nodes)
        A.add_edges(edges)
        nx.set_node_attributes(A.core_network, {c['node_for_adding']: c for c in nodes})

        result.append(compute_attributes(A, ['system', 'complex', 'polymer', 'monomer'])[0])
    return pd.DataFrame(result)


# filtered_with_network_measures=mhc_filtered.pipe(calculate_measures)


# filtered_with_network_measures.to_csv('D:/study/rit/DSCI.799.01-GraduateCapstone/HLAPeptideInteraction/datasets/cedar/raw/filtered_with_network_measures.csv',index=False)
import pytorch_lightning as pl


def modify_allele_name(x):
    return ''.join(x.split(':')[0:1]) + ':' + ''.join(x.split(':')[1:2])


class NetworkMeasuresDataModule(pl.LightningDataModule):
    def __init__(self,
                 tcell_table_file,
                 hlalleles_prot_fastas_seq,
                 allellist_file, transformation_function, batch_size=32,
                 data_dir='cedar/processed/',
                 patient_one_file_name='PatientOne.xlsx',
                 patient_two_file_name='PatientTwo.xlsx',
                 file_name="tcr_filtered_with_network_measures.csv",
                 type='train', filter_criteria={'Epitope:Parent Species': 'Homo sapiens'},
                 IEDB_response_code={'Positive': 1,
                                     'Positive-High': 1,
                                     'Positive-Intermediate': 1,
                                     'Positive-Low': 1,
                                     'Negative': 0,
                                     },
                 root_folder='datasets/cedar/',
                 cell_tissue_type=None,
                 dest_name="tcr_filtered_with_network_measures.csv"):
        super().__init__()
        self.prepare_data_per_node = True
        self.data_dir = data_dir
        self.file_name = file_name
        self.type = type
        self.tcell_table_file = tcell_table_file
        self.hlalleles_prot_fastas_seq = hlalleles_prot_fastas_seq
        self.allellist_file = allellist_file
        self.IEDB_response_code = IEDB_response_code
        self.filter_criteria = filter_criteria
        self.transformation_function = transformation_function
        self.root_folder = root_folder
        self.dest_name = dest_name
        self.batch_size = batch_size
        self.patient_one_file_name = patient_one_file_name
        self.patient_two_file_name = patient_two_file_name
        self.patient_one_alleles = ['B*57:01', 'A*11:01:01', 'A*03:01:01', 'B*15:32:01', 'B*13:01:01', 'C*07:02:01',
                                    'C*12:03:0']
        self.cell_tissue_type=cell_tissue_type

    def prepare_data(self):
        if not os.path.isdir(os.path.join(self.root_folder, 'processed')):
            os.mkdir(os.path.join(self.root_folder, 'processed'))
        if not os.path.isfile(os.path.join(self.root_folder, 'processed', self.dest_name)):
            allellist = pd.read_csv(self.allellist_file, sep=",")
            allellist = allellist.astype({column: 'string' for column in allellist.columns})
            fasta_sequences = SeqIO.parse(open(self.hlalleles_prot_fastas_seq), 'fasta')
            alles_fastea_seq_df = pd.DataFrame(
                [{'AlleleID': str(fasta.id), "Description": fasta.description, "Sequence": str(fasta.seq)} for fasta in
                 fasta_sequences])
            alles_fastea_seq_df = alles_fastea_seq_df.astype(
                {column: 'string' for column in alles_fastea_seq_df.columns})
            alles_fastea_seq_df['AlleleID'] = alles_fastea_seq_df['AlleleID'].apply(lambda x: x.replace('HLA:', ''))

            allellist = allellist.merge(alles_fastea_seq_df, on='AlleleID')
            allellist = allellist.drop_duplicates()
            allellist['Sequence'] = allellist['Sequence'].str.strip()

            with open(self.tcell_table_file, 'r', encoding='utf-8') as input_handle:
                csv_reader = csv.reader(input_handle, delimiter=',')
                header_1 = next(csv_reader)
                header_2 = next(csv_reader)
                header_list = []
                for x, y in zip(header_1, header_2):
                    header_list.append(':'.join([x, y]))

            mhc_full = pd.read_csv(self.tcell_table_file, skiprows=2)
            mhc_full.columns = header_list
            mhc_full['Allele'] = mhc_full['MHC:Allele Name'].apply(lambda x: str(x).replace("HLA-", ''))
            mhc_full['Epitope:Description'] = mhc_full['Epitope:Description'].str.strip()
            idx = mhc_full['Epitope:Description'].str.contains(' ')
            mhc_full.loc[idx, 'Epitope:Description'] = mhc_full.loc[idx, 'Epitope:Description'].apply(
                lambda x: x[0:x.index(' ')])

            mhc_full['AlleleOrig'] = mhc_full.Allele
            allellist['AlleleOrig'] = allellist.Allele
            mhc_full['Allele'] = mhc_full.Allele.apply(
                lambda x: ''.join(x.split(':')[0:1]) + ':' + ''.join(x.split(':')[1:2]))
            allellist['Allele'] = allellist.Allele.apply(
                lambda x: ''.join(x.split(':')[0:1]) + ':' + ''.join(x.split(':')[1:2]))

            # matched_alleles = mhc_full[
            #     mhc_full.Allele.isin([modify_allele_name(al) for al in self.patient_one_alleles])].Allele.unique()
            # missing_alleles = [modify_allele_name(al) for al in self.patient_one_alleles if
            #                    modify_allele_name(al) not in matched_alleles]

            mhc_filtered = mhc_full[(mhc_full['Allele'].isin(allellist.Allele))]


            mhc_filtered['Sequence'] = mhc_filtered.Allele.apply(
                lambda x: allellist.loc[allellist.Allele == x, 'Sequence'].unique()[0])

            mhc_filtered.rename(columns={"Description": "Allele:Description", "Sequence": "Allele:Sequence"})
            mhc_filtered['y'] = mhc_filtered['Assay:Qualitative Measure'].apply(lambda x: self.IEDB_response_code[x])

            filter=(mhc_filtered['Epitope:Parent Species'] == 'Homo sapiens')     & (mhc_filtered['Epitope:Description'].str.len() <= 25)    & (mhc_filtered['Epitope:Description'].str.len() >= 8) & (mhc_filtered['MHC:Class'] == 'I')
            if self.cell_tissue_type is not None:
                filter =(mhc_filtered['Epitope:Parent Species'] == 'Homo sapiens')     & (mhc_filtered['Epitope:Description'].str.len() <= 25)    & (mhc_filtered['Epitope:Description'].str.len() >= 8) & (mhc_filtered['MHC:Class'] == 'I') & (mhc_filtered['Antigen Presenting Cells:Cell Tissue Type'] == self.cell_tissue_type)
            mhc_filtered = mhc_filtered[filter]
            filtered_with_network_measures = mhc_filtered.pipe(self.transformation_function)
            filtered_with_network_measures.to_csv(os.path.join(self.root_folder, 'processed', self.dest_name),
                                                  index=False)
        if not os.path.isfile(
                os.path.join(self.root_folder, 'processed', self.patient_one_file_name.replace('.xlsx', '.csv'))):
            allellist = pd.read_csv(self.allellist_file, sep=",")
            allellist = allellist.astype({column: 'string' for column in allellist.columns})
            fasta_sequences = SeqIO.parse(open(self.hlalleles_prot_fastas_seq), 'fasta')
            alles_fastea_seq_df = pd.DataFrame(
                [{'AlleleID': str(fasta.id), "Description": fasta.description, "Sequence": str(fasta.seq)} for fasta in
                 fasta_sequences])
            alles_fastea_seq_df = alles_fastea_seq_df.astype(
                {column: 'string' for column in alles_fastea_seq_df.columns})
            alles_fastea_seq_df['AlleleID'] = alles_fastea_seq_df['AlleleID'].apply(lambda x: x.replace('HLA:', ''))

            allellist = allellist.merge(alles_fastea_seq_df, on='AlleleID')
            allellist = allellist.drop_duplicates()
            allellist['Sequence'] = allellist['Sequence'].str.strip()
            patient_one_df = pd.read_excel(os.path.join(self.root_folder, 'raw', self.patient_one_file_name),
                                           engine='openpyxl')
            patient_one_df['y'] = pd.cut(patient_one_df['y'], 2, include_lowest=True, labels=[0, 1], )
            patient_one_df['Epitope:Description'] = patient_one_df['Epitope:Description'].str.strip()
            idx = patient_one_df['Epitope:Description'].str.contains(' ')
            patient_one_df.loc[idx, 'Epitope:Description'] = patient_one_df.loc[idx, 'Epitope:Description'].apply(
                lambda x: x[0:x.index(' ')])

            allellist['AlleleOrig'] = allellist.Allele
            patient_one_df['AlleleOrig'] = patient_one_df['Allele']

            patient_one_df['Allele'] = patient_one_df.Allele.apply(
                lambda x: ''.join(x.split(':')[0:1]) + ':' + ''.join(x.split(':')[1:2]))
            allellist['Allele'] = allellist.Allele.apply(
                lambda x: ''.join(x.split(':')[0:1]) + ':' + ''.join(x.split(':')[1:2]))

            patient_one_df = patient_one_df[(patient_one_df['Allele'].isin(allellist.Allele))]
            # patient_one_df = patient_one_df.merge(allellist, on='Allele')
            patient_one_df['Sequence'] = patient_one_df.Allele.apply(
                lambda x: allellist.loc[allellist.Allele == x, 'Sequence'].unique()[0])
            patient_one = patient_one_df.pipe(self.transformation_function)
            patient_one.to_csv(
                os.path.join(self.root_folder, 'processed', self.patient_one_file_name.replace('xlsx', 'csv')),
                index=False)

    def setup(self, stage: str):
        if stage == "fit":
            torch.manual_seed(0)
            dataset = NetworkMeasuresDataset(dir=self.data_dir, file_name=self.file_name)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size

            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        if stage == "test":
            torch.manual_seed(0)
            dataset = NetworkMeasuresDataset(dir=self.data_dir, file_name=self.file_name)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size

            _, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


        if stage == "predict":
            self.predict_dataset = NetworkMeasuresDataset(dir=self.data_dir, file_name=self.file_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)


class NetworkMeasuresDataset(Dataset):

    def __init__(self, dir, file_name, resample=True):
        self.data = pd.read_csv(os.path.join(dir, file_name))

        # self.data=self.data.groupby('matched_pbmc')
        # self.data = self.data.apply(lambda x: x.sample(self.data.size().max(),replace=True).reset_index(drop=True))
        ros = RandomOverSampler(random_state=1)
        self.y = self.data.pop('matched_pbmc')
        self.x = self.data
        if resample:
            self.x, self.y = ros.fit_resample(self.x, self.y)
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.data['matched_pbmc'], test_size=0.2, random_state=1)
        # self.X_train.pop('matched_pbmc')
        # self.X_test.pop('matched_pbmc')
        self.Y = torch.LongTensor(np.array(self.y))
        self.X = torch.Tensor(np.array(self.x))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LitUnet(pl.LightningModule):
    def __init__(self, model, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1_train = F1Score(task='binary')
        self.auroc_train = AUROC(task='binary')

        self.valid_acc = torchmetrics.Accuracy(task='binary')
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.f1_valid = F1Score(task='binary')
        self.auroc_val = AUROC(task='binary')


        self.f1_test = F1Score(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_auroc = AUROC(task='binary')


        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.accuracy(torch.argmax(y_pred, -1), y)
        self.f1_train(torch.argmax(y_pred, -1), y)
        self.auroc_train(torch.argmax(y_pred, -1), y)
        # self.log('train_acc_step', self.accuracy, on_step=True, on_epoch=True)
        # self.log('train_f1_step', self.f1_train, on_step=True, on_epoch=True)
        # self.log("train_loss", loss, on_step=True, on_epoch=True)
        # self.log('train_auroc', self.accuracy, on_step=True, on_epoch=True)
        self.log_dict(
            {'train_loss': loss, "train_f1": self.f1_train, 'train_auroc': self.auroc_train,
             'train_acc': self.accuracy}, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.model(x)
        return torch.argmax(y_pred, -1), y

    def test_epoch_end(self, output_results):
        # this out is now the full size of the batch
        y_pred=torch.cat([y_pred for y_pred, _ in output_results])
        y=torch.cat([y for _, y in output_results])
        self.f1_test(y_pred,y)
        self.test_acc(y_pred,y)
        self.test_precision(y_pred,y)
        self.test_recall(y_pred,y)
        self.test_auroc(y_pred,y)

        self.log_dict({
          'test_f1': self.f1_test,
          'test_acc': self.test_acc,
          'test_auroc': self.test_auroc,
          'test_precision': self.test_precision,
          'test_recall': self.test_recall}, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.valid_acc(torch.argmax(y_pred, -1), y)
        self.f1_valid(torch.argmax(y_pred, -1), y)
        self.auroc_val(torch.argmax(y_pred, -1), y)
        self.val_precision(torch.argmax(y_pred, -1), y)
        self.val_recall(torch.argmax(y_pred, -1), y)

        # self.log("valid_loss", loss)
        # self.log('valid_f1_step', self.f1_valid, on_step=True, on_epoch=True)
        # self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        # self.log('valid_auroc', self.auroc_val, on_step=True, on_epoch=True)
        self.log_dict({'val_loss': loss,
                       'val_f1': self.f1_valid,
                       'val_acc': self.valid_acc,
                       'val_auroc': self.auroc_val,
                       'val_precision': self.val_precision,
                       'val_recall': self.val_recall},
                      on_step=True, on_epoch=True)
        return loss

    # def training_epoch_end(self, outs):
    #     self.log_dict(
    #         { "train_f1": self.f1_train, 'train_auroc': self.auroc_train, 'train_acc': self.accuracy},
    #         on_step=False, on_epoch=True)
    #     # log epoch metric
    #     # self.log('train_acc_epoch', self.accuracy)
    #     # self.log('valid_acc_epoch', self.valid_acc)
    #     # self.log ('train_f1_epoch', self.f1_train)
    #     # self.log('valid_f1_epoch', self.f1_valid)
    #     #
    #     # self.log("performance_training_epoch", {"train_f1": self.f1_train, 'train_auroc': self.auroc_train,"train_acc":self.accuracy,}, on_step=False, on_epoch=True)
    #     # self.log("performance_valid_epoch",
    #     #          {"val_f1": self.f1_valid, 'val_auroc': self.auroc_val, "val_acc": self.valid_acc, }, on_step=True, on_epoch=False)
    #
    # def validation_epoch_end(self, outs) -> None:
    #     self.log_dict(
    #         { "val_f1": self.f1_valid, 'val_acc': self.valid_acc, 'val_auroc': self.auroc_val},
    #         on_step=False, on_epoch=True)
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UnetAtt(torch.nn.Module):
    def calc_accuracy(self, Y_Pred: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Get the accuracy with respect to the most likely label
        :param Y_Pred:
        :param Y:
        :return:
        """

        # return the values & indices with the largest value in the dimension where the scores for each class is
        # get the scores with largest values & their corresponding idx (so the class that is most likely)
        max_scores, max_idx_class = Y_Pred.max(
            dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
        # usually 0th coordinate is batch size
        n = Y.size(0)
        assert (n == max_idx_class.size(0))
        # calulate acc (note .item() to do float division)
        acc = (max_idx_class == Y).sum().item() / n
        return acc

    def __init__(self):
        super().__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm1d(22, affine=False)
        self.conv = nn.Conv1d(22, 18, 1)
        self.conv1 = nn.Conv1d(18, 16, 1)
        self.conv2 = nn.Conv1d(16, 8, 1)
        self.deconv1 = nn.ConvTranspose1d(8, 16, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.deconv2 = nn.ConvTranspose1d(16, 18, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(18)
        self.deconv3 = nn.ConvTranspose1d(32, 64, kernel_size=1)
        self.deconv4 = nn.ConvTranspose1d(36, 240, kernel_size=1)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.attent1 = Attention_block(16, 8, 16)
        self.attent2 = Attention_block(18, 16, 32)
        self.attent3 = Attention_block(64, 18, 64)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.215)
        self.dense1 = nn.Linear(240, 120)
        self.batch_normf1 = nn.BatchNorm1d(120, affine=False)
        self.dense2 = nn.Linear(120, 60)
        self.batch_normf2 = nn.BatchNorm1d(60, affine=False)
        self.dense3 = nn.Linear(60, 30)
        self.batch_normf3 = nn.BatchNorm1d(30, affine=False)
        self.dense4 = nn.Linear(30, 2)

    def forward(self, x):
        x = torch.reshape(x, list(x.shape) + [-1])
        x = self.batch_norm(x)
        x1 = self.conv(x)
        x1 = self.Maxpool(x1)

        x2 = self.conv1(x1)
        x2 = self.Maxpool(x2)

        x3 = self.conv2(x2)
        x3 = self.Maxpool(x3)

        # decoder start

        x4 = self.relu(self.deconv1(x3))
        x4 = self.batch_norm1(x4)
        x4 = self.attent1(x4, x3)
        x5 = torch.cat([x3, x4], dim=1)

        x6 = self.relu(self.deconv2(x5))
        x6 = self.batch_norm2(x6)
        x6 = self.attent2(x6, x2)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.relu(self.deconv3(x7))
        x8 = self.batch_norm3(x8)
        x8 = self.attent3(x8, x1)
        x9 = torch.cat([x8, x1], dim=1)
        x9 = self.deconv4(x9)
        x10 = torch.reshape(x9, (x9.shape[0], -1))
        x10 = self.relu(self.Maxpool(x10))

        x10 = self.dropout(x10)
        x10 = self.batch_normf1(F.sigmoid(self.dense1(x10)))
        x10 = self.batch_normf2(F.sigmoid(self.dense2(x10)))
        x10 = self.batch_normf3(F.sigmoid(self.dense3(x10)))
        x10 = F.sigmoid(self.dense4(x10))

        return x10


# read training data
# training_data_file = "training_data_copy.csv"
# training_pair_list = []
# peptides_with_unknown_aa = 0
# allele_freq_dict = {}
# with open(training_data_file, 'r') as input_handle:
#     csv_reader = csv.DictReader(input_handle, delimiter=',')
#     for row in csv_reader:
#         peptide = row['Epitope:Description']
#         peptide = list(peptide)
#         peptide_assertion = True
#         for aa in peptide:
#             if aa not in aa_list:
#                 peptide_assertion = False
#                 break
#         if not peptide_assertion:
#             peptides_with_unknown_aa += 1
#             continue
#         response = IEDB_response_code[row['Assay:Qualitative Measure']]
#         allele = row['MHC:Allele Name']
#         if allele in allele_freq_dict:
#             allele_freq_dict[allele] += 1
#         else:
#             allele_freq_dict[allele] = 1
#         training_pair_list.append([peptide, response])
#
# print("allele_freq_dict =", allele_freq_dict)
# print("peptides_with_unknown_aa = ", peptides_with_unknown_aa)
# print("len(training_pair_list) = ", len(training_pair_list))
# print("  positive = ", len([y for x, y in training_pair_list if y == 1]))
# print("  negative = ", len([y for x, y in training_pair_list if y == 0]))
# print("training_pair_list[0] = ", training_pair_list[0])
# training_data=pd.read_csv(training_data_file)
# training_data['y']=training_data['Assay:Qualitative Measure'].apply(lambda x: IEDB_response_code[x])
#
# normal_hla_file = 'D:/study/rit/DSCI.799.01-GraduateCapstone/HLAPeptideInteraction/datasets/deepImmun/raw/' + patient + "_normal_hla.txt"  # "mel15_normal_hla.txt"
# with open(normal_hla_file, 'r') as input_handle:
#     normal_hla = input_handle.readlines()
#     normal_hla = [x.strip() for x in normal_hla]
# normal_hla_neg = normal_hla
#
# normal_hla_neg_product_df=pd.DataFrame([ [None, None,hla,None,None,petide,list(IEDB_response_code.keys())[list(IEDB_response_code.values()).index(0)],0] for hla,petide in itertools.product(hla_alleles, normal_hla)],columns=training_data.columns)
#
# training_data_merged_neg_hla=training_data.append(normal_hla_neg_product_df, ignore_index=True)
# for id, group in training_data.groupby(['Reference:Reference ID','y']):
#    print(group.loc[:,['MHC:Allele Name','Epitope:Description','y']])
output_file = "training_data_copy.csv"
tcell_table_file = 'tcell_full_v3.csv'
hlalleles_prot_fastas_seq = 'hla_prot.fasta'
allellist_file = 'Allelelist.txt'
dm = NetworkMeasuresDataModule(tcell_table_file=tcell_table_file, hlalleles_prot_fastas_seq=hlalleles_prot_fastas_seq,
                               allellist_file=allellist_file, transformation_function=calculate_measures,
                               batch_size=512)

dm.prepare_data()
dm.setup(stage="fit")
model_version = 'tcr_all'
models_dir='trained_models_task_fine_tune'
tensorboard = pl_loggers.TensorBoardLogger("tb_logs", name="UnetAttTask", version=model_version)


checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(os.path.abspath(os.path.dirname(__file__)), models_dir, model_version), verbose=True,
    save_top_k=1, filename=model_version)
trainer = pl.Trainer(
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
    accelerator='gpu',
    devices=[0],
    logger=tensorboard, max_epochs=20000)

model = LitUnet(model=UnetAtt(), learning_rate=0.01)

if os.path.isfile(os.path.join(os.path.abspath(os.path.dirname(__file__)), models_dir, model_version,
                               model_version + '.ckpt')):

    trainer.fit(model=LitUnet(model=UnetAtt(), learning_rate=0.01), datamodule=dm,
                ckpt_path=os.path.join(os.path.abspath(os.path.dirname(__file__)),models_dir, model_version,
                                       model_version + '.ckpt'))
else:
    trainer.fit(model=LitUnet(model=UnetAtt(), learning_rate=0.01), datamodule=dm)

trainer.test(model=LitUnet(model=UnetAtt(), learning_rate=0.01), datamodule=dm,
             ckpt_path=os.path.join(os.path.abspath(os.path.dirname(__file__)),models_dir, model_version,
                                    model_version + '.ckpt'))
