# Transformers
The initial transformer was fine tuned using ProtienBERT as the base model:  
https://huggingface.co/Rostlab/prot_bert

The initial training scheme was to use 5-fold cross validation with stratified splits. The final model incorperated all the training data with the best hyperparameters from CV 5.
The semi-supervised model was trained on psuedo-labels from the 5-fold cross validated transformer. The entire psuedo-labeled corpus can be found in Raw_64M_Labels_Scores.csv.  

## Hypertuning Scripts
HT_CV_2.py - Contains the 5-fold cross validation for the ProteinBert transformer. (DL_Pytorch)  
To run: `python HT_CV_2.py`  

semi-supervised_learning_transformer.py - Generates psuedo-labels, prunes based on model confidence, and trains new model for one epoch. (DL_Pytorch)  
To run: `python semi_supervised_learning_transformer.py [base_model]`  

## Plotting Scripts
extract_model_embeddings.py - Returns the embeddings of a model. (DL_Pytorch)  
To run: `python extract_model_embeddings.py [model]`  

visualize_model_embeddings.py - Uses UMAP on one percent of the peptide manifold to transform the WALTZDB database into embeddings space. (RAPIDS)  
To run: `python visualize_model_embeddings.py`  

generate_roc_and_hist_plot.py - Generates ROC plot for specifiecd models. Also makes classification histogram for both models based on model confidence. (DL_Pytorch)   
To run: `python generate_roc_and_hist_plot.py [model_1] [model_2] [hist_name_1] [hist_name_2] [roc_plot_name]`  

prospective_predictions.py - Creates preditictions from the semi-supervised transformer for the 64M peptides. (DL_Pytorch)  
To run: `python prospective_predictions.py [model]`  
