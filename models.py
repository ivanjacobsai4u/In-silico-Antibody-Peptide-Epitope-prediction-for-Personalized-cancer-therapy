import torch
import torch.nn.functional as F
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours, \
    RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.nn import Linear, ModuleList
from torch_geometric.nn import HeteroConv, TransformerConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, GraphConv

import findseq
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import transforms
from imblearn.over_sampling import RandomOverSampler


class TopologicalAttributesDataset(Dataset):

    def __init__(self, file_name, type='train',sampling_strat=1):
        self.type = type
        df = pd.read_csv(file_name, delimiter=',')

        df = df.loc[:, (df != 0).any(axis=0)]
        all = df.copy()
        # bins = [-1000, 1, 30, 100, 500, 1000, 10000];
        # self.class_names = np.array(['none', 'very_weak', 'weak', 'strong', 'very_strong', 'extremely_strong', 'mega_strong']);
        # matched_pbmc_all = all.pop('matched_pbmc')
        # # Discretize continuous PBMC values to classes
        # all_classes = np.digitize(matched_pbmc_all.to_numpy(), bins, right=True)
        # all['class'] = all_classes
        matched_pbmc_all = all.pop('matched_pbmc')
        all['class']=pd.cut(matched_pbmc_all,2, include_lowest=True,labels=[0, 1],)
        self.class_names = np.array(['weak', 'strong']);
        if sampling_strat==1:
            all=all.groupby('class')
            all=all.apply(lambda x: x.sample(all.size().max(), replace=True).reset_index(drop=True))
            X_train, X_test, y_train, y_test=train_test_split(all, all['class'], test_size=0.2, random_state=1)
            y_train = X_train.pop('class')
            y_test = X_test.pop('class')
        if sampling_strat == 2:
            X_train, X_test, y_train, y_test = train_test_split(all, all['class'], test_size=0.2, random_state=1)
            # X_train = X_train.groupby('class')
            # # Balance data to have equal number of classes
            # X_train = X_train.apply(lambda x: x.sample(X_train.size().max(), replace=True).reset_index(drop=True))
            y_train = X_train.pop('class')
            ros = RandomOverSampler(random_state=1)
            cc = ClusterCentroids(random_state=1)
            rus = RandomUnderSampler(random_state=1,replacement=True)
            nm1 = NearMiss(version=2)
            cnn = CondensedNearestNeighbour(random_state=0)

            X_train, y_train = ros.fit_resample(X_train, y_train)
            # X_test = X_test.groupby('class')
            # Balance data to have equal number of classes
            # X_test = X_test.apply(lambda x: x.sample(X_test.size().max(), replace=True).reset_index(drop=True))
            y_test = X_test.pop('class')


        if self.type == "train":
            self.x, self.y= X_train,y_train
        if self.type == "test":
            self.x, self.y = X_test,y_test

        self.X = torch.Tensor(np.array(self.x))
        self.Y = torch.LongTensor(np.array(self.y))

    def get_all_data(self, dl_model):
        if dl_model:
            return self.X, self.Y
        else:
            return self.x, self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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
        self.batch_norm = nn.BatchNorm1d(20, affine=False)
        self.conv = nn.Conv1d(20, 18, 1)
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
        self.batch_norm = nn.BatchNorm1d(20, affine=False)

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


class ShallowNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(20, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, 120)
        self.fc5 = nn.Linear(120, 2)

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
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class ModelFactory:

    def make_model(self, model_type):
        return self._init_implementation(model_type)

    def _init_implementation(self, model_type):
        type_to_implementation = {
            'rf': RandomForestClassifier,
            'sgd': SGDClassifier,
            'gnb': GaussianNB,
            'lrc': LogisticRegression,
            'mlp': MLPClassifier,
            'cnb': ComplementNB,
            'gbc': GradientBoostingClassifier,
            'knnCl': KNeighborsClassifier,
            'treeCL': DecisionTreeClassifier,
            "shNet": ShallowNet,
            'UnetAtt': UnetAtt

        }
        implementation = type_to_implementation.get(model_type, None)
        return implementation()


class HeteroGNN(torch.nn.Module):
    def __init__(self, nr_classes, layers, hidden=32):
        super().__init__()
        self.aminoAcids = list(set(findseq.one_letter.values()))
        self.aminoEmbedding = torch.nn.Embedding(len(set(list(findseq.one_letter.values()))) + 1, hidden)
        self.atomEmbedding = torch.nn.Embedding(109, hidden)
        self.hlaEmbedding = torch.nn.Embedding(300, hidden)
        self.hidden = hidden
        self.convs = []

        for _ in range(layers):
            m_layers = dict()
            m_layers['atom'] = HeteroConv({('atom', 'has_bond', 'atom'): TransformerConv((-1, -1), hidden),
                                           ('atom', 'has_polar_contact', 'atom'): TransformerConv((-1, -1),
                                                                                                  hidden),
                                           ('coord', 'rev_has', 'atom'): TransformerConv((-1, -1), hidden)},
                                          aggr='mean')
            m_layers['monomer'] = HeteroConv({
                ('atom', 'rev_has', 'monomer'): TransformerConv((-1, -1), hidden),
                ('monomer', 'chains_to', 'monomer'): TransformerConv((-1, -1), hidden),
                ('monomer', 'has_polar_contact', 'monomer'): TransformerConv((-1, -1), hidden), }, aggr='mean')
            m_layers['polymer'] = HeteroConv({
                ('monomer', 'rev_has', 'polymer'): TransformerConv((-1, -1), hidden),
            }, aggr='mean')
            m_layers['complex'] = HeteroConv({
                ('polymer', 'part_of', 'complex'): TransformerConv((-1, -1), hidden),
            }, aggr='mean')
            m_layers['system'] = HeteroConv({
                ('complex', 'part_of', 'system'): TransformerConv((-1, -1), hidden),
            }, aggr='mean')
            self.convs.append(m_layers)

        self.lin = torch.nn.Linear(hidden, 1)
        self.linCls = torch.nn.Linear(hidden, nr_classes)

        self.softmax = torch.nn.Softmax(-1)
        self.jk = JumpingKnowledge(mode="lstm", channels=hidden, num_layers=layers)

        self.bn1 = torch.nn.BatchNorm1d(hidden)

    def get_emb_amino_one_letter(self, amino):

        return self.aminoEmbedding(amino)

    def get_emb_atom(self, atom):
        return self.atomEmbedding(atom)

    def forward(self, x_dict, edge_index_dict):

        xs = [x_dict['system']]
        for m_layers in self.convs:
            x_dict['atom'] = F.leaky_relu(self.bn1(m_layers['atom'](x_dict, edge_index_dict)['atom']))
            x_dict['monomer'] = F.leaky_relu(self.bn1(m_layers['monomer'](x_dict, edge_index_dict)['monomer']))
            x_dict['polymer'] = F.leaky_relu(self.bn1(m_layers['polymer'](x_dict, edge_index_dict)['polymer']))
            x_dict['complex'] = F.leaky_relu(self.bn1(m_layers['complex'](x_dict, edge_index_dict)['complex']))
            x_dict['system'] = F.leaky_relu(self.bn1(m_layers['system'](x_dict, edge_index_dict)['system']))
            xs += [x_dict['system']]

        x = self.jk(xs)

        xcls = F.log_softmax(self.linCls(x), -1)
        x_regr = self.lin(x)

        return x_regr, xcls
