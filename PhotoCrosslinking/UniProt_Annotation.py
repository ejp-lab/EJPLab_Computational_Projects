import pandas as pd
import numpy as np
import os
import sys
import urllib
from bs4 import BeautifulSoup
from tqdm import tqdm

protein_list = sys.argv[1]

def get_uniprot(query, query_type):
    #query is uniprotid or pdbcode, query type is 'ACC' or 'PBD_ID'
    url = 'https://www.uniprot.org/uploadlists/'
    params = {'from':query_type, 'to':'ACC', 'format':'txt', 'query':query}

    data = urllib.parse.urlencode(params)
    data = data.encode('ascii')
    request = urllib.request.Request(url, data)
    with urllib.request.urlopen(request) as response:
        res = response.read()
        page = BeautifulSoup(res, features = 'lxml').get_text()
        page = page.splitlines()

    return page

protein_df = pd.read_csv(protein_list, index_col = 0)

for index, row in tqdm(protein_df.iterrows()):
    uniprot_acc = row['Fasta headers'].split()[0].split('|')[1]
    uniprot_data = get_uniprot(uniprot_acc, 'ACC')
    GO_localization_list = []
    GO_function_list = []
    GO_process_list = []
    domain_list = []
    repeat_list = []
    for count, line in enumerate(uniprot_data):
        if 'DR   GO;' in line:
            GO_line = line.replace('DR   GO; GO:', '').replace(';', '').split(':')
            if 'C' in GO_line[0]:
                GO_localization_list.append(GO_line[1])
            elif 'F' in GO_line[0]:
                GO_function_list.append(GO_line[1])
            elif 'P' in GO_line[0]:
                GO_process_list.append(GO_line[1])
            else:
                pass
        elif 'FT   DOMAIN' in line:
            domain_id = uniprot_data[count + 1].split()
            domain_id = ' '.join(domain_id[1:]).replace('/note="', '').replace('"', '')
            domain_location = line.split()
            domain_location = ' '.join(domain_location[2:]).replace('FT DOMAIN ', '')
            domain_info = domain_id + ' {' + domain_location + '}'
            domain_list.append(domain_info)
        elif 'FT   REPEAT' in line:
            repeat_id = uniprot_data[count + 1].split()
            repeat_id = ' '.join(repeat_id[1:]).replace('/note="', '').replace('"', '')
            repeat_location = line.split()
            repeat_location = ' '.join(repeat_location[2:]).replace('FT REPEAT ', '')
            repeat_info = repeat_id + ' {' + repeat_location + '}'
            repeat_list.append(repeat_info)
        else:
            pass

    protein_df.loc[index, 'GO Localization'] = (', '.join(list(set(GO_localization_list))))
    protein_df.loc[index, 'GO Function'] = (', '.join(list(set(GO_function_list))))
    protein_df.loc[index, 'GO Process'] = (', '.join(list(set(GO_process_list))))
    protein_df.loc[index, 'Domains'] = ', '.join(domain_list)
    protein_df.loc[index, 'Repeats'] = ', '.join(repeat_list)

protein_df.to_csv('annotated_' + protein_list)

