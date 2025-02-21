"""Utility functions."""
import os
import functools
import logging
import pickle
import hashlib

import wandb
from openai import OpenAI
from tenacity import (retry, wait_random_exponential)

import numpy as np
import scipy
from sklearn import metrics

CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
api = wandb.Api()
api.entity = os.environ['WANDB_API_ENTITY']

MAJOR = 'Major error'
def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data, ), function, n_resamples=n_resamples, confidence_level=0.9,
            random_state=rng)
        return {
            'std_err': bs.standard_error,
            'low': bs.confidence_interval.low,
            'high': bs.confidence_interval.high
        }
    return inner


def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)


def accuracy_at_quantile(accuracies, uncertainties, quantile):
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    return np.mean(accuracies[select])


def area_under_thresholded_accuracy(accuracies, uncertainties):
    quantiles = np.linspace(0.1, 1, 20)
    select_accuracies = np.array([accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles])
    dx = quantiles[1] - quantiles[0]
    area = (select_accuracies * dx).sum()
    return area


# Need wrappers because scipy expects 1D data.
def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        # this function is called in the bootstrap
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        return bootstrap(helper, rng=rng)(y_true_y_score)
    return converted_func


def wandb_restore(wandb_run, filename):
    files_dir = 'tmp_wandb/'
    os.system(f'rm -rf {files_dir}')
    os.system(f'mkdir -p {files_dir}')

    run = api.run(wandb_run)
    run.file(filename).download(
        root=files_dir, replace=True, exist_ok=False)
    with open(f'{files_dir}/{filename}', 'rb') as f:
        out = pickle.load(f)
    return out, run.config


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG


def log_w_indent(text, indent, symbol='>>'):
    """Log and add indent."""
    ind2col = {i: f"\x1b[{a}" for i, a in enumerate([
        '1m', '31m', '33m', '34m', '35m'])}
    reset = "\x1b[0m"

    if indent > 0:
        logging.info(
            ind2col[indent] + (indent * 2) * symbol + ' ' + text + reset)
    else:
        logging.info(ind2col[indent] + text + reset)


def get_sentences(response):
    """Extract sentences from response."""
    # Some manual exceptions for sentence extraction.
    facts = response.replace('Ph.D.', 'PhD').replace('\n', ' ').split('. ')
    facts = [f.strip() + '.' if i < len(facts) - 1 else f.strip() for i, f in enumerate(facts)]
    for i in facts:
        print(i)


@retry(wait=wait_random_exponential(min=1, max=10))
def oai_predict(prompt):
    """Predict with GPT-4 model."""

    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    output = CLIENT.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        max_tokens=200,
    )
    response = output.choices[0].message.content
    return response


def predict_w_log(prompt, indent):
    """Predict and log inputs and outputs."""
    log_w_indent(f'Input: {prompt}', indent)
    response = oai_predict(prompt)
    log_w_indent(f'Output: {response}', indent, symbol='xx')
    return response


def get_metrics(all_labels, all_uncertainties):
    all_labels = np.array(all_labels)
    all_uncertainties = np.array(all_uncertainties)

    rng = np.random.default_rng(41)
    # Base accuracy of GPT-4 propositions correctly.
    out = dict(uncertainties=all_uncertainties)

    for wrongness in ['major_only', 'minor_and_major']:
        out[wrongness] = dict()

        if wrongness == 'major_only':
            # Only MAJOR becomes False.
            labels = [True if lab != MAJOR else False for lab in all_labels]
        elif wrongness == 'minor_and_major':
            # Only True is True. Both MINOR and MAJOR map to false.
            labels = [True if lab == 'True' else False for lab in all_labels]
        else:
            raise ValueError
        labels = np.array(labels)
        #print(labels)
        #pdb.set_trace()
        #assert set(labels) == {True, False} or len(set(labels))
        out[wrongness]['per_question'] = dict(labels=labels)

        out[wrongness]['performance'] = dict(
            mean=np.mean(labels),
            bootstrap=bootstrap(np.mean, rng)(labels))

        # Next, evaluate mean and bootstrap estimates of other metrics.

        eval_metrics = dict(zip(
            ['auroc', 'area_under_thresholded_accuracy', 'mean_uncertainty'],
            list(zip(
                [auroc, area_under_thresholded_accuracy, np.mean],
                [compatible_bootstrap, compatible_bootstrap, bootstrap]
            )),
        ))

        answer_fractions = [0.8, 0.9, 0.95]
        for answer_fraction in answer_fractions:
            key = f'accuracy_at_{answer_fraction}_answer_fraction'
            eval_metrics[key] = [
                functools.partial(accuracy_at_quantile, quantile=answer_fraction),
                compatible_bootstrap]

        fargs = {
            'auroc': [1 - labels, all_uncertainties],
            'area_under_thresholded_accuracy': [labels, all_uncertainties],
            'mean_uncertainty': [all_uncertainties]}
        for answer_fraction in answer_fractions:
            fargs[f'accuracy_at_{answer_fraction}_answer_fraction'] = [labels, all_uncertainties]

        out[wrongness]['uncertainty'] = {}
        for fname, (function, bs_function) in eval_metrics.items():
            metric_i = function(*fargs[fname])
            logging.info("%s: %f", fname, metric_i)
            out[wrongness]['uncertainty'][fname] = {
                'mean': metric_i,
                'bootstrap': bs_function(function, rng)(*fargs[fname])
            }
    return out


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


def extract_questions(gen_questions):

    compatibility = (
        os.environ['HALLU_RESTORE_ID'] in ['hallu_long/5yfel47n', 'hallu_long/rok13nf2'] and
        'gen_qs' in os.environ['HALLU_RESTORE_STAGES'])

    questions = []
    for i, q in enumerate(gen_questions.split('\n')):
        if q.startswith(f'{i + 1}. '):
            questions.append(q[3:])
        else:
            if not compatibility:
                questions.append(q)
            else:
                questions.append(q[3:])

    return questions


def get_yes_no(response):
    binary_response = response.lower()[:10]
    if 'yes' in binary_response:
        uncertainty = 0
    elif 'no' in binary_response:
        uncertainty = 1
    else:
        uncertainty = 1
        logging.warning('MANUAL NO!')
    return uncertainty
