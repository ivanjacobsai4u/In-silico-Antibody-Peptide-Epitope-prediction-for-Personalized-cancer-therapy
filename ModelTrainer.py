from pprint import pprint
import json

from numpy import ndarray
from torch.utils.data import DataLoader
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, f1_score

from models import ModelFactory, TopologicalAttributesDataset
from sklearn.metrics import precision_recall_curve, precision_score
from sklearn.metrics import auc


class ModelTrainer(object):
    def __init__(self, type_to_implementation, names_models,
                 verbosity=logging.INFO,
                 csv_path_patient_1="all_systems_attributes_patient_1.csv",
                 csv_path_patient_2="all_systems_attributes_patient_2.csv"):
        self.type_to_implementation = type_to_implementation
        self.names_models = names_models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter()
        self.model_factory = ModelFactory()
        self.csv_path_patient_1 = csv_path_patient_1
        self.csv_path_patient_2 = csv_path_patient_2

        self.top_attr_patient_one_train = TopologicalAttributesDataset(self.csv_path_patient_1,sampling_strat=1)
        self.train_loader_patient_one = DataLoader(self.top_attr_patient_one_train, batch_size=256, shuffle=True);
        self.top_attr_patient_one_test = TopologicalAttributesDataset(self.csv_path_patient_1, type='test',sampling_strat=1)
        self.test_loader_patient_one = DataLoader(self.top_attr_patient_one_test, batch_size=256, shuffle=True);

        self.top_attr_patient_two_train = TopologicalAttributesDataset(self.csv_path_patient_2,sampling_strat=2)
        self.train_loader_patient_two = DataLoader(self.top_attr_patient_two_train, batch_size=256, shuffle=True);
        self.top_attr_patient_two_test = TopologicalAttributesDataset(self.csv_path_patient_2, type='test',sampling_strat=2)
        self.test_loader_patient_two = DataLoader(self.top_attr_patient_two_test, batch_size=256, shuffle=True);

        logging.basicConfig(filename='training.log', level=verbosity)

    def _train_cnn_model(self, model, nr_epochs, train_loader, test_loader, model_type):
        # Writer will output to ./runs/ directory by default

        model = model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()


        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(nr_epochs):
            running_loss = 0.0
            running_acc = 0.0
            running_loss_test = 0.0
            running_acc_test = 0.0

            for i, (train, test) in enumerate(zip(train_loader, test_loader)):
                train_data, y_train = train
                test_data, y_test = test
                test_data = test_data.to(self.device)
                train_data = train_data.to(self.device)

                y_train = y_train.to(self.device)
                y_test = y_test.to(self.device)

                optimizer.zero_grad()
                y_pred = model(train_data)

                # Compute and print loss
                loss = criterion(y_pred, y_train)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                y_pred_test = model(test_data)
                loss_test = criterion(y_pred_test, y_test)
                # print statistics
                running_loss += loss.item()
                running_loss_test += loss_test.item()
                running_acc += model.calc_accuracy(y_pred, y_train)
                running_acc_test += model.calc_accuracy(y_pred_test, y_test)
                if i % 10 == 0:  # print every 100 mini-batches

                    logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                    logging.info(f'[{epoch + 1}, {i + 1:5d}] accuracy: {running_acc / 10:.3f}')
                    self.writer.add_scalar('Loss/train {}'.format(model_type), running_loss / 10, epoch + 1)
                    self.writer.add_scalar('Accuracy/train {}'.format(model_type), running_acc / 10, epoch + 1)
                    self.writer.add_scalar('Loss/test {}'.format(model_type), running_loss_test / 10, epoch + 1)
                    self.writer.add_scalar('Accuracy/test {}'.format(model_type), running_acc_test / 10, epoch + 1)
                    running_loss = 0.0
                    running_acc = 0.0
                    running_loss_test = 0.0
                    running_acc_test = 0.0
        return model

    def _calc_metrics(self, y_pred_test, y_true_test):

        accuracy = accuracy_score(y_true_test, y_pred_test)
        precision, recall, thresholds = precision_recall_curve(y_true_test, y_pred_test)
        precision_sc = precision_score(y_true_test, y_pred_test, average='weighted')
        recall_sc = recall_score(y_true_test, y_pred_test, average='weighted')
        area_under_the_curve = auc(recall, precision)
        f1_sc = f1_score(y_true_test, y_pred_test, average='weighted')
        metrics = {'accuracy': accuracy, 'precision': precision_sc, 'recall': recall_sc, "f1_sc": f1_sc,'area_under_the_curve':area_under_the_curve}
        for k in metrics.keys():
            metrics[k] = round(metrics[k], 2)
        return metrics

    def _train_sklearn(self, model, train_data):
        X, y = train_data
        model.fit(X, y)
        return model

    def train(self, nr_epochs, persist_results=True):
        train_results = {}
        predictions = {}
        for model_type in self.type_to_implementation:
            model = self.model_factory.make_model(model_type)
            logging.info(model_type)

            if  isinstance(model,torch.nn.Module):
                model_patient_one = self._train_cnn_model(model, nr_epochs, self.train_loader_patient_one, self.test_loader_patient_one, model_type)
                X_test_one, y_true_test_one = self.top_attr_patient_one_test.get_all_data(dl_model=isinstance(model_patient_one,torch.nn.Module))
                model_patient_two = self._train_cnn_model(model, nr_epochs, self.train_loader_patient_two, self.test_loader_patient_two, model_type)
                X_test_two, y_true_test_two = self.top_attr_patient_two_test.get_all_data(dl_model=isinstance(model_patient_two, torch.nn.Module))
                X_test_two = X_test_two.to(self.device)
                X_test_one = X_test_one.to(self.device)


                y_pred_test_one = model_patient_one(X_test_one)
                y_pred_test_one = torch.argmax(y_pred_test_one, -1)

                y_pred_test_m_one_data_pat_two = model_patient_one(X_test_two)
                y_pred_test_m_one_data_pat_two = torch.argmax(y_pred_test_m_one_data_pat_two, -1)

                metrics = self._calc_metrics(y_pred_test_one.cpu().detach().numpy(), y_true_test_one.cpu().detach().numpy())
                metrics_p_one_data_p_two = self._calc_metrics(y_pred_test_m_one_data_pat_two.cpu().detach().numpy(),
                                             y_true_test_two.cpu().detach().numpy())
                train_results[self.names_models[model_type]+ "_patientOne"] = {'metrics': metrics}
                train_results[self.names_models[model_type] + "_patientOne_on_data_p2"] = {'metrics': metrics_p_one_data_p_two}
                predictions[self.names_models[model_type]+ "_patientOne"] = {'y_pred_test': y_pred_test_one.cpu().detach().numpy().tolist(),'y_true_test': y_true_test_one.cpu().detach().numpy().tolist()}
                predictions[self.names_models[model_type] + "_patientOne_on_data_p2"] = {       'y_pred_test': y_pred_test_m_one_data_pat_two.cpu().detach().numpy().tolist(),
                    'y_true_test': y_true_test_two.cpu().detach().numpy().tolist()}




                y_pred_test_two = model_patient_two(X_test_two)
                y_pred_test_two = torch.argmax(y_pred_test_two, -1)

                y_pred_test_two_data_p1 = model_patient_two(X_test_one)
                y_pred_test_two_data_p1 = torch.argmax(y_pred_test_two_data_p1, -1)

                metrics = self._calc_metrics(y_pred_test_two.cpu().detach().numpy(), y_true_test_two.cpu().detach().numpy())
                metrics_mp2_data_p1 = self._calc_metrics(y_pred_test_two_data_p1.cpu().detach().numpy(),
                                             y_true_test_one.cpu().detach().numpy())
                train_results[self.names_models[model_type] + "_patientTwo"] = {'metrics': metrics}
                train_results[self.names_models[model_type] + "_patientTwo_on_data_p1"] = {'metrics': metrics_mp2_data_p1}
                predictions[self.names_models[model_type]   + "_patientTwo"] = {'y_pred_test': y_pred_test_two.cpu().detach().numpy().tolist(),
                                                                                'y_true_test': y_true_test_two.cpu().detach().numpy().tolist()}
                predictions[self.names_models[model_type] + "_patientTwo_on_data_p1"] = {
                    'y_pred_test': y_pred_test_two_data_p1.cpu().detach().numpy().tolist(),
                    'y_true_test': y_true_test_one.cpu().detach().numpy().tolist()}

            else:
                model_patient_one = self._train_sklearn(model, self.top_attr_patient_one_train.get_all_data(dl_model=isinstance(model, torch.nn.Module)))
                X_test, y_true_test = self.top_attr_patient_one_test.get_all_data(dl_model=isinstance(model,torch.nn.Module))
                y_pred_test = model_patient_one.predict(X_test)

                metrics = self._calc_metrics(y_pred_test, y_true_test)

                train_results[self.names_models[model_type] + "_patientOne"] = {'metrics': metrics}
                predictions[self.names_models[model_type] + "_patientOne"] = {'y_pred_test': y_pred_test,
                                                                              'y_true_test': y_true_test}

                model_patient_two = self._train_sklearn(model, self.top_attr_patient_two_train.get_all_data(dl_model=isinstance(model,torch.nn.Module)))
                X_test, y_true_test = self.top_attr_patient_two_test.get_all_data(dl_model=isinstance(model,torch.nn.Module))
                y_pred_test = model_patient_two.predict(X_test)

                metrics = self._calc_metrics(y_pred_test, y_true_test)

                train_results[self.names_models[model_type] + "_patientTwo"] = {'metrics': metrics}
                predictions[self.names_models[model_type] + "_patientTwo"] = {'y_pred_test': y_pred_test,
                                                                              'y_true_test': y_true_test}
            logging.info(metrics)
        if persist_results:
            with open('train_results_{}_epochs.json'.format(str(nr_epochs)), 'w') as fp:
                json.dump(train_results, fp)

        return train_results, predictions
