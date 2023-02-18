import logging
import warnings

warnings.filterwarnings("ignore")
from ModelTrainer import ModelTrainer
import json
import numpy as np
from mlxtend.evaluate import mcnemar, mcnemar_table
import itertools
import scipy.stats as stats

with open('configs/configs.json', "r") as f:
    configs = json.load(f)

names_models = configs['statistics']["names_models"]
type_to_implementation = configs['statistics']["type_to_implementation"]
csv_path_patient_1 = configs['statistics']["csv_path_patient_1"]
csv_path_patient_2 = configs['statistics']["csv_path_patient_2"]

nr_epochs = configs['statistics']['nr_epochs']
model_trainer = ModelTrainer(names_models=names_models,
                             type_to_implementation=type_to_implementation,
                             csv_path_patient_1=csv_path_patient_1,
                             csv_path_patient_2=csv_path_patient_2)
train_merge_results = {}
for model_name in names_models.values():
    train_merge_results[model_name + "_patientOne"] = {'y_pred_test': [], 'y_true_test': []}
    train_merge_results[model_name + "_patientTwo"] = {'y_pred_test': [], 'y_true_test': []}

train_merge_results['UnetAtt_patientOne_on_data_p2']= {'y_pred_test': [], 'y_true_test': []}
train_merge_results['UnetAtt_patientTwo_on_data_p1']= {'y_pred_test': [], 'y_true_test': []}
model_pairs = list(itertools.combinations(names_models.values(), 2))
odds_results = {}
for pair in model_pairs:
    model_one, model_two = pair
    odds_results[model_one + '_' + model_two + '_patientOne'] = {'oddsratios': [], 'pvalues': []}
    odds_results[model_one + '_' + model_two + '_patientTwo'] = {'oddsratios': [], 'pvalues': []}




for n in range(configs['statistics']['nr_iterations']):
    _, train_results = model_trainer.train(nr_epochs=nr_epochs, persist_results=False)
    logging.info(train_results)
    for model_name in names_models.values():
        train_merge_results[model_name + "_patientOne"]['y_pred_test'].extend(
            train_results[model_name + "_patientOne"]['y_pred_test'])
        train_merge_results[model_name + "_patientOne"]['y_true_test'].extend(
            train_results[model_name + "_patientOne"]['y_true_test'])
        train_merge_results[model_name + "_patientTwo"]['y_pred_test'].extend(
            train_results[model_name + "_patientTwo"]['y_pred_test'])
        train_merge_results[model_name + "_patientTwo"]['y_true_test'].extend(
            train_results[model_name + "_patientTwo"]['y_true_test'])

    train_merge_results['UnetAtt_patientOne_on_data_p2']['y_pred_test'].extend(train_results['UnetAtt_patientOne_on_data_p2']['y_pred_test'])
    train_merge_results['UnetAtt_patientOne_on_data_p2']['y_true_test'].extend(
        train_results['UnetAtt_patientOne_on_data_p2']['y_true_test'])

    train_merge_results['UnetAtt_patientTwo_on_data_p1']['y_pred_test'].extend(
        train_results['UnetAtt_patientTwo_on_data_p1']['y_pred_test'])
    train_merge_results['UnetAtt_patientTwo_on_data_p1']['y_true_test'].extend(
        train_results['UnetAtt_patientTwo_on_data_p1']['y_true_test'])
    for pair in model_pairs:
        model_one, model_two = pair
        y_target = np.array(train_results[model_one + "_patientOne"]['y_true_test'])
        y_model1 = np.array(train_results[model_one + "_patientOne"]['y_pred_test'])
        y_model2 = np.array(train_results[model_two + "_patientOne"]['y_pred_test'])
        tb = mcnemar_table(y_target=y_target,
                        y_model1=y_model1,
                            y_model2=y_model2)
        # chi2, p = mcnemar(ary=tb, exact=True, corrected=True)
        res= stats.barnard_exact(tb, alternative="less")
        oddsratio, pvalue=res.statistic,res.pvalue
        odds_results[model_one + '_' + model_two + "_patientOne"]['oddsratios'].append(oddsratio)
        odds_results[model_one + '_' + model_two + "_patientOne"]['pvalues'].append(pvalue)

        y_target_patient_two = np.array(train_results[model_one + "_patientTwo"]['y_true_test'])
        y_model1_patient_two = np.array(train_results[model_one + "_patientTwo"]['y_pred_test'])
        y_model2_patient_two = np.array(train_results[model_two + "_patientTwo"]['y_pred_test'])
        tb = mcnemar_table(y_target=y_target_patient_two,y_model1=y_model1_patient_two,y_model2=y_model2_patient_two)
        # chi2, p = mcnemar(ary=tb, exact=True, corrected=True)
        res = stats.barnard_exact(tb, alternative="less")
        oddsratio, pvalue = res.statistic, res.pvalue
        odds_results[model_one + '_' + model_two + "_patientTwo"]['oddsratios'].append(oddsratio)
        odds_results[model_one + '_' + model_two + "_patientTwo"]['pvalues'].append(pvalue)
stats_results = {}
for pair in model_pairs:
    model_one, model_two = pair
    y_target = np.array(train_merge_results[model_one + "_patientOne"]['y_true_test'])
    y_model1 = np.array(train_merge_results[model_one + "_patientOne"]['y_pred_test'])
    y_model2 = np.array(train_merge_results[model_two + "_patientOne"]['y_pred_test'])
    tb = mcnemar_table(y_target=y_target,
                       y_model1=y_model1,
                       y_model2=y_model2)
    chi2, p = mcnemar(ary=tb, corrected=True)
    # oddsratio,pvalue=stats.fisher_exact(tb)
    stats_results[model_one + '_' + model_two + "_patientOne"] = {'model_1_patientOne': model_one + "_patientOne",
                                                                  'model_2_patientOne': model_two + "_patientOne",
                                                                  'model_pairs': pair,
                                                                  'chi-squared': chi2, 'p-value': round(p, 5),
                                                                  'oddsratio': round(np.min(np.array(
                                                                      odds_results[model_one + '_' + model_two+'_patientOne'][
                                                                          'oddsratios'])), 2),
                                                                  'oddsratios':
                                                                      odds_results[model_one + '_' + model_two+'_patientOne'][
                                                                          'oddsratios'],
                                                                  'pvalue': round(np.min(np.array(
                                                                      odds_results[model_one + '_' + model_two+'_patientOne'][
                                                                          'pvalues'])), 2)}

    y_target = np.array(train_merge_results[model_one + "_patientTwo"]['y_true_test'])
    y_model1 = np.array(train_merge_results[model_one + "_patientTwo"]['y_pred_test'])
    y_model2 = np.array(train_merge_results[model_two + "_patientTwo"]['y_pred_test'])
    tb = mcnemar_table(y_target=y_target,
                       y_model1=y_model1,
                       y_model2=y_model2)
    chi2, p = mcnemar(ary=tb, corrected=True)
    # oddsratio,pvalue=stats.fisher_exact(tb)
    stats_results[model_one + '_' + model_two + "_patientTwo"] = {'model_1_patientTwo': model_one + "_patientTwo",
                                                                  'model_2_patientTwo': model_two + "_patientTwo",
                                                                  'model_pairs': pair,
                                                                  'chi-squared': chi2, 'p-value': round(p, 5),
                                                                  'oddsratio': round(np.min(np.array(odds_results[model_one + '_' + model_two + "_patientTwo"]['oddsratios'])), 2),
                                                                  'oddsratios':odds_results[model_one + '_' + model_two + "_patientTwo"]['oddsratios'],
                                                                  'pvalue': round(np.min(np.array(odds_results[model_one + '_' + model_two + "_patientTwo"][
                                                                          'pvalues'])), 2)}

y_target = np.array(train_merge_results['UnetAtt_patientTwo']['y_true_test'])
y_model2 = np.array(train_merge_results["UnetAtt_patientOne_on_data_p2"]['y_pred_test'])
y_model1 = np.array(train_merge_results["UnetAtt_patientTwo"]['y_pred_test'])
tb = mcnemar_table(y_target=y_target,
                        y_model1=y_model1,
                            y_model2=y_model2)
chi2, p = mcnemar(ary=tb, exact=True, corrected=True)
res= stats.barnard_exact(tb, alternative="less")
oddsratio, pvalue=res.statistic,res.pvalue

stats_results['UnetAtt_patientTwo_UnetAtt_patientOne_on_data_p2']={'p-value':round(p, 5),'oddsratio':oddsratio}


y_target = np.array(train_merge_results['UnetAtt_patientOne']['y_true_test'])
y_model2 = np.array(train_merge_results["UnetAtt_patientTwo_on_data_p1"]['y_pred_test'])
y_model1 = np.array(train_merge_results["UnetAtt_patientOne"]['y_pred_test'])
tb = mcnemar_table(y_target=y_target,
                        y_model1=y_model1,
                            y_model2=y_model2)
chi2, p = mcnemar(ary=tb, exact=True, corrected=True)
res= stats.barnard_exact(tb, alternative="less")
oddsratio, pvalue=res.statistic,res.pvalue

stats_results['UnetAtt_patientOne_UUnetAtt_patientTwo_on_data_p1']={'p-value':round(p, 5),'oddsratio':oddsratio}


with open('statistics_results_{}_epochs_{}_iterations.json'.format(str(nr_epochs),
                                                                   str(configs['statistics']['nr_iterations'])),
          'w') as fp:
    json.dump(stats_results, fp)

# with open('preds_stats_results_{}_epochs_{}_iterations.json'.format(str(nr_epochs),
#                                                                     str(configs['statistics'][
#                                                                             'nr_iterations'])),
#           'w') as fp:
#     json.dump(train_merge_results, fp)

logging.info('finished')
print("finished")
