import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.init as initialization_strategy
from transformers import T5EncoderModel
import utils


utils.seed_everything()

class PerSequenceRegressionLoraDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, mutname, wtname, labelname, n_labels):
        self.tokenizer = tokenizer
        self.dataframe = dataframe.copy()
        self.mut = self.dataframe[mutname]
        self.wt = self.dataframe[wtname]
        self.labels = self.dataframe[labelname]
        self.max_len = max_len
        self.mutname = mutname
        self.wtname = wtname
        self.n_labels = n_labels

    def __len__(self):
        return len(self.mut)

    def __getitem__(self, index):

        mut = self.mut[index]
        wt = self.wt[index]
        label = self.labels[index]

        inputs = self.tokenizer([mut, wt], padding='max_length', max_length=self.max_len, truncation=True) 
        ids, mask = inputs['input_ids'], inputs['attention_mask']

        return {'input_ids': torch.tensor(ids, dtype=torch.int), 'attention_mask': torch.tensor(mask, dtype=torch.int), 'labels': torch.tensor(label, dtype=torch.bfloat16)}

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()

        utils.seed_everything()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        
        self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
        self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))

        self.multi_lora_a = nn.Parameter(torch.ones(self.scaling_rank, linear_layer.in_features) + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale)
        self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    def forward(self, input):

        weight = self.weight
        weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
        weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank

        return F.linear(input, weight, self.bias)

class PerSequenceRegressionLora(torch.nn.Module):

    def __init__(self, hyperparameters_dictionary, tokenizer):
        super(PerSequenceRegressionLora, self).__init__()

        utils.seed_everything()

        self.embedding_shape = 1024
        self.out_shape = hyperparameters_dictionary['out_shape']
        self.dropout_weight = hyperparameters_dictionary['dropout_weight']
        self.lora_rank = hyperparameters_dictionary['lora_rank']
        self.lora_scaling_rank = hyperparameters_dictionary['lora_scaling_rank']
        self.lora_init_scale = hyperparameters_dictionary['lora_init_scale']
        self.lora_modules = hyperparameters_dictionary['lora_modules']
        self.lora_layers = hyperparameters_dictionary['lora_layers']
        self.lora_trainable_param_names = hyperparameters_dictionary['lora_trainable_param_names']

        self.encoder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.encoder.resize_token_embeddings(len(tokenizer))
        for m_name, module in dict(self.encoder.named_modules()).items():
            if re.fullmatch(self.lora_modules, m_name):
                for c_name, layer in dict(module.named_children()).items():
                    if re.fullmatch(self.lora_layers, c_name):
                        assert isinstance(layer, nn.Linear), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                        setattr(module, c_name, LoRALinear(layer, self.lora_rank, self.lora_scaling_rank, self.lora_init_scale))
        
        for (param_name, param) in self.encoder.shared.named_parameters():
            param.requires_grad = False
        for (param_name, param) in self.encoder.encoder.named_parameters():
            param.requires_grad = False       

        for (param_name, param) in self.encoder.named_parameters():
            if re.fullmatch(self.lora_trainable_param_names, param_name):
                param.requires_grad = True

        self.dropout_layer = torch.nn.Dropout(self.dropout_weight)
        self.activation_function = F.relu

        self.l1 = torch.nn.Linear(self.embedding_shape, 512, bias=True)
        self.l2 = torch.nn.Linear(512, 256, bias=True)
        self.l3 = torch.nn.Linear(256, 128, bias=True)
        self.output_layer = torch.nn.Linear(128, self.out_shape, bias=True)
        
        initialization_strategy.xavier_uniform_(self.l1.weight)
        initialization_strategy.xavier_uniform_(self.l2.weight)
        initialization_strategy.xavier_uniform_(self.l3.weight)
        initialization_strategy.xavier_uniform_(self.output_layer.weight)
        
        self.l1.bias.data.fill_(0.01)
        self.l2.bias.data.fill_(0.01)
        self.l3.bias.data.fill_(0.01)
        self.output_layer.bias.data.fill_(0.01)

    def forward(self, ids, mask):

        utils.seed_everything()
        
        id_mut = ids[:, 0]
        id_wt = ids[:, 1]
        mask_mut = mask[:, 0]
        mask_wt = mask[:, 1]

        encoder_mut = torch.mean(self.encoder(input_ids=id_mut, attention_mask=mask_mut).last_hidden_state, dim=1)
        encoder_wt = torch.mean(self.encoder(input_ids=id_wt, attention_mask=mask_wt).last_hidden_state, dim=1)
        encoder_delta = encoder_mut - encoder_wt
        
        logits = self.l1(encoder_delta)
        logits = self.activation_function(logits)
        logits = self.dropout_layer(logits)
        logits = self.l2(logits)
        logits = self.activation_function(logits)
        logits = self.dropout_layer(logits)
        logits = self.l3(logits)
        logits = self.activation_function(logits)
        
        output = self.output_layer(logits)

        return output