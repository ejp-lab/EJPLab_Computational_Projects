# Instructions to Machine Learn Features of Interest

In an interactive python session execute the following command:\
    `run Preprocessing.py feature_file.csv number_of_splits number_of_stratified_repeats`
    
After optimal features have been selected, grid searching can be performed to derive the most suitable hyperparameters for each model.\
    `run ExampleHypertuning.py feature_file_with_selected_features.csv number_of_splits number_of_stratified_repeats`
