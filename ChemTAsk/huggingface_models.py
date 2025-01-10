"""Implement HuggingfaceModel models."""
# Modified script to include ChatGPT logprobs
import copy
import logging
from collections import Counter
import torch

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download

from openai import OpenAI
import os
from math import log10

from abc import ABC, abstractmethod
from typing import List, Text


STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']


class BaseModel(ABC):

    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        pass


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.token_limit = 7900 if model_name == 'llama' else 100000

        if stop_sequences == 'default':
            self.stop_sequences = STOP_SEQUENCES

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        if 'llama' in model_name.lower():

            if '70b' in model_name.lower():
                model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
            
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16, max_memory={0: '80GIB'})
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, device_map='auto', token_type_ids=None)

        if 'gpt' in model_name.lower():

            self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, device_map='auto', token_type_ids=None)

    def predict(self, input_data, temperature, return_full=False):

        # Implement prediction.
        if 'llama' in self.model_name.lower():
            inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

            if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
                if 'token_type_ids' in inputs:  # Some HF models have changed.
                    del inputs['token_type_ids']
                pad_token_id = self.tokenizer.eos_token_id
            else:
                pad_token_id = None

            if self.stop_sequences is not None:
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                    stops=self.stop_sequences,
                    initial_length=len(inputs['input_ids'][0]),
                    tokenizer=self.tokenizer)])
            else:
                stopping_criteria = None

            logging.debug('temperature: %f', temperature)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    do_sample=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=pad_token_id,
                )

            if len(outputs.sequences[0]) > self.token_limit:
                raise ValueError(
                    'Generation exceeding token limit %d > %d',
                    len(outputs.sequences[0]), self.token_limit)

            full_answer = self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=True)

            if return_full:
                return full_answer

            # For some models, we need to remove the input_data from the answer.
            if full_answer.startswith(input_data):
                input_data_offset = len(input_data)
            else:
                # This can happen if something is out of the tokenizers vocab
                # Ex: structure.Itmay -> structure .Itmay
                # Something like ure.It is not in there


                raise ValueError('Have not tested this in a while.')

            # Remove input from answer.
            answer = full_answer[input_data_offset:]

            # Remove stop_words from answer.
            stop_at = len(answer)
            sliced_answer = answer
            if self.stop_sequences is not None:
                for stop in self.stop_sequences:
                    if answer.endswith(stop):
                        stop_at = len(answer) - len(stop)
                        sliced_answer = answer[:stop_at]
                        break
                if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                    error_msg = 'Error: Stop words not removed successfully!'
                    error_msg += f'Answer: >{answer}< '
                    error_msg += f'Sliced Answer: >{sliced_answer}<'
                    if 'falcon' not in self.model_name.lower():
                        raise ValueError(error_msg)
                    else:
                        logging.error(error_msg)

            # Remove whitespaces from answer (in particular from beginning.)
            sliced_answer = sliced_answer.strip()

            # Get the number of tokens until the stop word comes up.
            # Note: Indexing with `stop_at` already excludes the stop_token.
            # Note: It's important we do this with full answer, since there might be
            # non-trivial interactions between the input_data and generated part
            # in tokenization (particularly around whitespaces.)
            token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
            n_input_token = len(inputs['input_ids'][0])
            n_generated = token_stop_index - n_input_token

            if n_generated == 0:
                logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
                n_generated = 1

            # Get the last hidden state (last layer) and the last token's embedding of the answer.
            # Note: We do not want this to be the stop token.

            # outputs.hidden_state is a tuple of len = n_generated_tokens.
            # The first hidden state is for the input tokens and is of shape
            #     (n_layers) x (batch_size, input_size, hidden_size).
            # (Note this includes the first generated token!)
            # The remaining hidden states are for the remaining generated tokens and is of shape
            #    (n_layers) x (batch_size, 1, hidden_size).

            # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size).
            # We do not get embeddings for input_data! We thus subtract the n_tokens_in_input from
            # token_stop_index to arrive at the right output.

            if 'decoder_hidden_states' in outputs.keys():
                hidden = outputs.decoder_hidden_states
            else:
                hidden = outputs.hidden_states

            if len(hidden) == 1:
                """
                logging.warning(
                    'Taking first and only generation for hidden! '
                    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                    'last_token: %s, generation was: %s',
                    n_generated, n_input_token, token_stop_index,
                    self.tokenizer.decode(outputs['sequences'][0][-1]),
                    full_answer,
                    )
                """
                last_input = hidden[0]
            elif ((n_generated - 1) >= len(hidden)):
                # If access idx is larger/equal.
                logging.error(
                    'Taking last state because n_generated is too large'
                    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                    'last_token: %s, generation was: %s, slice_answer: %s',
                    n_generated, n_input_token, token_stop_index,
                    self.tokenizer.decode(outputs['sequences'][0][-1]),
                    full_answer, sliced_answer
                    )
                last_input = hidden[-1]
            else:
                last_input = hidden[n_generated - 1]

            # Then access last layer for input
            last_layer = last_input[-1]
            # Then access last token in input.
            last_token_embedding = last_layer[:, -1, :].cpu()

            # Get log_likelihoods.
            # outputs.scores are the logits for the generated token.
            # outputs.scores is a tuple of len = n_generated_tokens.
            # Each entry is shape (bs, vocabulary size).
            # outputs.sequences is the sequence of all tokens: input and generated.
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True)
            # Transition_scores[0] only contains the scores for the first generated tokens.
            
            log_likelihoods = [score.item() for score in transition_scores[0]]
            if len(log_likelihoods) == 1:
                #logging.warning('Taking first and only generation for log likelihood!')
                log_likelihoods = log_likelihoods
            else:
                log_likelihoods = log_likelihoods[:n_generated]

            if len(log_likelihoods) == self.max_new_tokens:
                pass
                #logging.warning('Generation interrupted by max_token limit.')

            if len(log_likelihoods) == 0:
                raise ValueError

            return sliced_answer, log_likelihoods, last_token_embedding
        
        else:

            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input_data}],
                temperature=temperature,
                logprobs=True,
                max_tokens=self.max_new_tokens
            )

            content = response.choices[0].message.content.strip()
            logprobs = response.choices[0].logprobs.content[-1].logprob

            return content, logprobs, torch.tensor([0])

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""


        if 'llama' in self.model_name.lower():
            input_data += ' A'
            
            tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']
            # The computation of the negative log likelihoods follows:
            # https://huggingface.co/docs/transformers/perplexity.

            target_ids_true = tokenized_prompt_true.clone()
            # Set all target_ids except the last one to -100.
            target_ids_true[0, :-1] = -100

            with torch.no_grad():
                model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

            loss_true = model_output_true.loss

            return -loss_true.item()
        
        else:
            
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input_data}],
                temperature=0,
                logprobs=True,
                max_tokens=1
            )
            content = response.choices[0].message.content.strip()
            logprobs = response.choices[0].logprobs.content[-1].logprob
            print(content)
            if "A" in content:
                return logprobs
            elif "B" in content:
                return log10(1 - (10 ** logprobs))
            else:
                return -10
