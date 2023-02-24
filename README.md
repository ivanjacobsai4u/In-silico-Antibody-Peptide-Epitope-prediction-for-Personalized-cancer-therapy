# In-silico-Antibody-Peptide-Epitope-prediction-for-Personalized-cancer-therapy
The human leukocyte antigen (HLA) system is a complex of genes on chromosome 6 in humans that encodes cell-surface proteins responsible for regulating the immune system. Viral peptides presented to cancer cell surfaces by the HLA trigger the immune system to kill the cells, creating Antibody-peptide epitopes (APE). This study proposes an in-silico approach to identify patient-specific APEs by applying complex networks diagnostics on a novel multiplex data structure as input for a deep learning model. The proposed analytical model identifies patient and tumor-specific APEs with as few as 20 labeled data points. Additionally, the proposed data structure employs complex network theory, and other statistical approaches that can better explain and reduce the black box effect of deep learning. The proposed approach achieves an F1-score above 80% on patient and tumor specific tasks and minimizes training time and the number of parameters.


### Data
To download the data visit:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7652206.svg)](https://doi.org/10.5281/zenodo.7652206)
Download and put in data folder in root of project.

HLA Alleles list [IPD-IMGT/HLA](https://www.ebi.ac.uk/ipd/imgt/hla/download/)


To train the personalized models run train_pipe.py
To generate statics run statistics_pipe.py
To train and evaluate tumor specific model run cedar_tcr.py for all tissues. Initialize the NetworkMeasuresDataModule with desired tissue e.g.,

`dm = NetworkMeasuresDataModule(tcell_table_file=tcell_table_file, 
                                hlalleles_prot_fastas_seq=hlalleles_prot_fastas_seq,
                                allellist_file=allellist_file, 
                                transformation_function=calculate_measures,
                                batch_size=512,cell_tissue_type='Lymphoid') `

Training produces tensorboard logs and trained models .ckpt in tb_logs and trained models folders respectively.
The folders are created and populated by the training procedure.
To visualize tensorboard logs run `tensorboard --logdir tb_logs` in cmd