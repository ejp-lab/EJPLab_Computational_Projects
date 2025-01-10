import pandas as pd
import numpy as np
import os
import sys
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import random
from scipy.stats import ttest_1samp as ttest
from scipy.stats import shapiro as shapiro
from sklearn.feature_selection import SelectFdr

file = sys.argv[1]

proteomic_df = pd.read_csv(file, sep = '\t')

proteomic_df_columns = proteomic_df.columns
HL_columns = []

#identify replicate columns for usage throughout the code
for column in proteomic_df.columns:
    if 'Ratio H/L ' in column:
        if 'variability' in column:
            pass
        elif 'normalized' in column:
            pass
        elif 'count' in column:
            pass
        elif 'type' in column:
            pass
        else:
            HL_columns.append(column)
    else:
        pass

filtered_df = proteomic_df

#remove contaminants
for index, row in filtered_df.iterrows():
    contaminant_test = row['Potential contaminant']
    if contaminant_test == '+':
        filtered_df = filtered_df.drop(index)
    else:
        pass

curated_len = len(filtered_df)
n_contaminants = len(proteomic_df) - curated_len
print(str(n_contaminants) + ' contaminants filtered out')

#remove reverse hits
for index, row in filtered_df.iterrows():
    reverse_test = row['Reverse']
    if reverse_test == '+':
        filtered_df = filtered_df.drop(index)
    else:
        pass

n_reverse_hits = curated_len - len(filtered_df)
print(str(n_reverse_hits) + ' reverse hits filtered out')
curated_len = len(filtered_df)

#remove rows with low H/L ratio quantification (i.e. all NaN)
for index, row in filtered_df.iterrows():
    NaN_count = 0
    for column in HL_columns:
        if str(row[column]) == 'nan':
            NaN_count = NaN_count + 1
        else:
            pass
    if NaN_count >= 0.7*len(HL_columns):
        # print('Protein identified containing no H/L quantification')
        # print('Protein ID is ' + row['Fasta headers'])
        # print('Exact NaN count is ' + str(NaN_count))
        filtered_df = filtered_df.drop(index)
    else:
        pass

#print some useful information about what you have filtered out of your data
n_no_quants = curated_len - len(filtered_df)
print(str(n_no_quants) + ' proteins identified lacking H/L quantification')
curated_len = len(filtered_df)
final_filter_loss = len(proteomic_df) - curated_len
print('\n')
print(str(final_filter_loss) + ' PROTEINS REMOVED IN TOTAL AFTER MINIMALIST FILTERING')

########perform dummy kNN to optimize imputation parameters#########

#calculate dataset nan percentage
test_df = filtered_df[HL_columns]
nan_number = test_df.isna().sum().sum()
non_nan_number = sum(test_df.count())
total_number = nan_number + non_nan_number
nan_percentage = nan_number/total_number
nan_perc_per_column = nan_percentage/len(HL_columns)

#prepare test_df with no NaNs
for index, row in test_df.iterrows():
    for column in HL_columns:
        if str(row[column]) == 'nan':
            test_df = test_df.drop(index)
            break
        else:
            pass

n_nan_proteins = len(filtered_df) - len(test_df)
print('\n')
print(str(n_nan_proteins) + ' proteins with NaNs temporarily filtered for kNN optimization')

dummy_df = test_df

#prepare dummy dataframe with missing values
dummy_df = dummy_df.mask(np.random.random(dummy_df.shape) < nan_percentage)

#prepare k for kNN and iterate over ks to optimize
k_range = range(1,21)
test_array = test_df.to_numpy()

for k in k_range: 
    imputer = KNNImputer(n_neighbors = k, weights = 'uniform', metric = 'nan_euclidean')
    imputer.fit(dummy_df)
    dummy_array = imputer.transform(dummy_df)
    rmse = mean_squared_error(test_array, dummy_array, squared = False)
    print('With a k of ' + str(k) + ' RMSE value is ' + str(rmse))
    if k == 1:
        minimum_rmse = rmse
    elif rmse < minimum_rmse:
        minimum_rmse = rmse
        optimal_k = k
    else:
        pass

print('\n')
print('Optimal k determined to be ' + str(optimal_k) + ' with an RMSE of ' + str(minimum_rmse))

#construct dataframes for kNN and conduct kNN imputation
trimmed_filtered_df = filtered_df[HL_columns]

imputer = KNNImputer(n_neighbors = optimal_k, weights = 'uniform', metric = 'nan_euclidean')
imputer.fit(trimmed_filtered_df)
trans_data = imputer.transform(trimmed_filtered_df)

trans_data_df = pd.DataFrame(trans_data, columns = ['H/L 1 imputed','H/L 2 imputed', 'H/L 3 imputed', 'H/L 4 imputed', 'H/L 5 imputed', 'H/L 6 imputed', 'H/L 7 imputed'])

final_dataframe = filtered_df.join(trans_data_df.set_index(filtered_df.index))
final_dataframe.to_csv('detailed_imputed.csv')

#cleanup the csv by getting rid of all the useless columns
all_columns = final_dataframe.columns
for column in all_columns:
    if 'Fasta headers' in column:
        pass
    elif column == 'Peptides':
        pass
    elif column == 'Sequence coverage [%]':
        pass
    elif column == 'Score':
        pass
    elif 'imputed' in column:
        pass
    elif column in HL_columns:
        pass
    else:
        final_dataframe = final_dataframe.drop(column, 1)

imputed_columns = []
for column in final_dataframe.columns:
    if 'imputed' in column:
        imputed_columns.append(column)
    else:
        pass

final_dataframe['Imputed Average'] = final_dataframe[imputed_columns].mean(axis = 1)

final_dataframe['Log2 Imputed H/L'] = np.log2(final_dataframe['Imputed Average'])

p_values = []

df_test = final_dataframe

for index, row in final_dataframe.iterrows():
    test_result = []
    pop_list = row[imputed_columns]

    tscore, pvalue = ttest(pop_list, popmean = 1)

    final_dataframe.loc[index, 'P-Value'] = pvalue

    test_result.append(index)
    test_result.append(pvalue)
    p_values.append(test_result)

final_dataframe['-Log10 P-value'] = np.log10(final_dataframe['P-Value']) * -1

final_dataframe.to_csv('simplified_imputed.csv')
