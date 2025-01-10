import os, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import shuffle
import utils, models
import optuna
import joblib
import time
import logging
import ipdb

def save_model(parameter_dictionary, outmodel_name, val_label_name, val_pred_name, test_label_name, test_pred_name):

    shutil_dir = trials_dir + 'Trial_' + str(parameter_dictionary['trial_num']) + '/'

    try:
        os.mkdir(shutil_dir)

    except:
        pass 

    shutil.move(outmodel_name, shutil_dir)
    shutil.move(val_label_name, shutil_dir)
    shutil.move(val_pred_name, shutil_dir)
    shutil.move(test_label_name, shutil_dir)
    shutil.move(test_pred_name, shutil_dir)

    return None

def eval_model(parameter_dictionary, model, device, trial_num, epoch, set_flag):

    t0 = time.time()
    tmp_params =  {'batch_size': parameter_dictionary['batch_size'], 'shuffle': False, 'num_workers': 0}
    
    if set_flag == 'Validation':
        tmp_loader = DataLoader(validation_set, **tmp_params)
    else:
        tmp_loader = DataLoader(testing_set, **tmp_params)
        
    len_tmp_dataloader = len(tmp_loader)
    tmp_loss_accumulator = 0

    model.eval()

    epoch_tmp_preds = []
    epoch_tmp_labels = []

    with torch.no_grad():
        for tmp_data_idx, tmp_data in enumerate(tmp_loader):

            tmp_percent_done = (((tmp_data_idx + 1) / len_tmp_dataloader) * 100)

            tmp_ids = tmp_data['input_ids'].to(device, dtype=torch.int)
            tmp_mask = tmp_data['attention_mask'].to(device, dtype=torch.int)
            tmp_labels = tmp_data['labels'].to(device, dtype=torch.bfloat16)

            tmp_outputs = model(tmp_ids, tmp_mask)

            tmp_loss, tmp_pred, tmp_truth = utils.loss_fn(tmp_outputs, tmp_labels, positive_weightings.to(device))
            tmp_loss_accumulator += tmp_loss.item()

            epoch_tmp_preds.extend(tmp_pred)
            epoch_tmp_labels.extend(tmp_truth)

            t_delta = time.time() - t0
            print(f'Trial: {trial_num}, {set_flag}: Epoch: {epoch}, Loss: {tmp_loss.item():.16f}, % Done: {tmp_percent_done:.5f}, Time Elapsed: {t_delta:.5f}')

    average_tmp_loss = np.round(tmp_loss_accumulator / len_tmp_dataloader, 8)

    return average_tmp_loss, epoch_tmp_labels, epoch_tmp_preds

def train_model(parameter_dictionary):

    trial_num = parameter_dictionary['trial_num']

    if trial_num <= number_jobs:
        study_best = 0
        
    else:
        study_best = study.best_trial.values[0]

    device = torch.device('cuda')

    train_params = {'batch_size': parameter_dictionary['batch_size'], 'shuffle': True, 'num_workers': 0}
    training_loader = DataLoader(training_set, **train_params)
    len_training_dataloader = len(training_loader)

    model = models.PerSequenceClassificationLoraPTEN(parameter_dictionary, tokenizer).bfloat16().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=parameter_dictionary['learning_rate'], fused=True)
    
    current_val_macro_f1 = 0
    outmodel_name = 'PTEN_' + str(trial_num) + '.pt'
    val_label_name = 'ValLabels.joblib'
    val_pred_name = 'ValPreds.joblib'
    test_label_name = 'TestLabels.joblib'
    test_pred_name = 'TestPreds.joblib'

    grad_accum_steps = parameter_dictionary['grad_accum_steps']
    patience_counter = 0
    patience = 15

    t0 = time.time()
    for epoch in range(parameter_dictionary['epochs']):

        epoch_train_preds = []
        epoch_train_labels = []

        model.train()

        train_loss_accumulator = 0

        optimizer.zero_grad()
        for train_data_idx, train_data in enumerate(training_loader):
         

            training_percent_done = (((train_data_idx + 1) / len_training_dataloader) * 100)

            train_ids = train_data['input_ids'].to(device, dtype=torch.int)
            train_mask = train_data['attention_mask'].to(device, dtype=torch.int)
            train_labels = train_data['labels'].to(device, dtype=torch.bfloat16)

            train_outputs = model(train_ids, train_mask)
            train_loss, train_pred, train_truth = utils.loss_fn(train_outputs, train_labels, positive_weightings.to(device))
            train_loss_accumulator += train_loss.item()
            
            train_loss.backward() # accumulate gradients

            epoch_train_preds.extend(train_pred)
            epoch_train_labels.extend(train_truth)

            t_delta = time.time() - t0
            print(f'Trial: {trial_num}, Training: Epoch: {epoch}, Loss: {train_loss.item():.16f}, % Done: {training_percent_done:.5f}, Time Elapsed: {t_delta:.5f}')

            if (train_data_idx + 1) % grad_accum_steps == 0 or (train_data_idx + 1) == len(training_loader):

                optimizer.step() # Update
                optimizer.zero_grad() # Clear

            if (train_data_idx + 1) % (grad_accum_steps * 10) == 0:
                print(classification_report(epoch_train_labels, epoch_train_preds))

                epoch_train_preds = []
                epoch_train_labels = []

            if (train_data_idx + 1) == int(np.floor(len_training_dataloader / 2)) or (train_data_idx + 1) == len_training_dataloader:

                average_train_loss = np.round(train_loss_accumulator / (train_data_idx + 1), 8)

                average_val_loss, epoch_val_labels, epoch_val_preds = eval_model(parameter_dictionary, model, device, trial_num, epoch, "Validation")
                new_val_macro_f1 = f1_score(epoch_val_labels, epoch_val_preds, average='macro')
                
                if new_val_macro_f1 >= current_val_macro_f1:

                    current_val_macro_f1 = new_val_macro_f1
                    patience_counter = 0

                    torch.save(model.state_dict(), outmodel_name)
                    joblib.dump(epoch_val_labels, val_label_name)
                    joblib.dump(epoch_val_preds, val_pred_name)

                else:
                    patience_counter += 1

                print('Average train loss: ' + str(average_train_loss))
                print('Average validation loss: ' + str(average_val_loss))
                print('Patience = ' + str(patience_counter))
                print()

                print('Validation Classification Report')
                print(classification_report(epoch_val_labels, epoch_val_preds))
                print()

                if patience_counter != patience:

                    model.train()

                else:

                    if current_val_macro_f1 >= study_best:

                        average_test_loss, epoch_test_labels, epoch_test_preds = eval_model(parameter_dictionary, model, device, trial_num, epoch, 'Testing')
                        print('Testing Metrics')
                        print(classification_report(epoch_test_labels, epoch_test_preds))

                        joblib.dump(epoch_test_labels, test_label_name)
                        joblib.dump(epoch_test_preds, test_pred_name)

                        save_model(parameter_dictionary, outmodel_name, val_label_name, val_pred_name, test_label_name, test_pred_name)

                    else:
                        os.remove(outmodel_name)
                        os.remove(val_label_name)
                        os.remove(val_pred_name)

                    model.to('cpu')
                    del model
                    torch.cuda.empty_cache()
                    return current_val_macro_f1
        
    average_test_loss, epoch_test_labels, epoch_test_preds = eval_model(parameter_dictionary, model, device, trial_num, epoch, 'Testing')

    print('Testing Metrics')
    print(classification_report(epoch_test_labels, epoch_test_preds))

    joblib.dump(epoch_test_labels, test_label_name)
    joblib.dump(epoch_test_preds, test_pred_name)

    save_model(parameter_dictionary, outmodel_name, val_label_name, val_pred_name, test_label_name, test_pred_name)
    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    return current_val_macro_f1

def objective(trial):

    torch.cuda.empty_cache()
    utils.seed_everything()
    trial_num = trial.number
    
    if trial_num != 0:
        best_params = study.best_trial.params
        best_params['best_trial_num'] = study.best_trial.number

        joblib.dump(best_params, 'best_hyperparameters.joblib')
        study.trials_dataframe().to_csv('optuna_data.csv', index=False)
        joblib.dump(study, 'optuna_study.joblib')

    parameters = {
            
            'in_shape' : PTEN_MAX_LEN,
            'out_shape' : N_LABELS,
            'epochs' : 50,
            'batch_size' : trial.suggest_int('batch_size', 2, 32, step=2),
            'grad_accum_steps' : trial.suggest_int('grad_accum_steps', 1, 16, step=1),
            'dropout_weight' : trial.suggest_float('dropout_weight', 0.1, 0.4, log=False),
            'learning_rate' : trial.suggest_float('learning_rate', 0.00001, 0.001, log=False),
            'trial_num' : trial_num,
            'lora_rank' : trial.suggest_int('lora_rank', 4, 8, step=1),
            'lora_scaling_rank' : 1,
            'lora_init_scale' : 0.01,
            'lora_modules' : '.*SelfAttention|.*EncDecAttention',
            'lora_layers' : 'q|k|v|o',
            'lora_trainable_param_names' : '.*layer_norm.*|.*lora_[ab].*'
            }

    print('\n')
    print(parameters)
    print()

    trial_loss = train_model(parameters)
    torch.cuda.empty_cache()

    return trial_loss

if __name__ == '__main__':
    cwd = os.getcwd() + '/'
    seed = 42

    utils.seed_everything(seed)
    utils.set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])
    trials_dir = utils.setup_trials_dir(cwd)

    NAME_COL = 'accession'
    SEQUENCE_COL = 'Mutated_Sequence_NO_HTL'
    LABEL_NAME = 'label'
    MASK_NAME = 'mask'
    PTEN_MAX_LEN = 410
    N_LABELS = 2

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", do_lower_case=False)

    train_df = shuffle(joblib.load('../../Data/PTEN_train.joblib'))
    val_df = shuffle(joblib.load('../../Data/PTEN_val.joblib'))
    test_df = shuffle(joblib.load('../../Data/PTEN_test.joblib'))
    
    train_df.index = train_df.index.astype(str)
    val_df.index = val_df.index.astype(str)
    test_df.index = test_df.index.astype(str)

    training_set = models.PerSequenceClassificationLoraDataset(train_df, tokenizer, PTEN_MAX_LEN, SEQUENCE_COL, LABEL_NAME, N_LABELS)
    validation_set = models.PerSequenceClassificationLoraDataset(val_df, tokenizer, PTEN_MAX_LEN, SEQUENCE_COL, LABEL_NAME, N_LABELS)
    testing_set = models.PerSequenceClassificationLoraDataset(test_df, tokenizer, PTEN_MAX_LEN, SEQUENCE_COL, LABEL_NAME, N_LABELS)

    positive_weightings = utils.get_pos_scalings(train_df, N_LABELS) 

    number_trials = 10
    number_jobs = 1

    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=number_trials, n_jobs=number_jobs, show_progress_bar=True)

    best_params = study.best_trial.params
    best_params['best_trial_num'] = study.best_trial.number

    joblib.dump(best_params, 'best_hyperparameters.joblib')
    study.trials_dataframe().to_csv('optuna_data.csv', index=False)
    joblib.dump(study, 'optuna_study.joblib')
