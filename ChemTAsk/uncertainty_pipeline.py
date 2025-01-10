"""Main script to run hallucination experiments."""
import os
import argparse

import pickle
from collections import defaultdict

import utils
import models
import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_factual_claims(response):

    # Prompts GPT4o to get the factual claims
    prompt = f"""Please list the specific factual propositions included in the answer below. Be complete and do not leave any factual claims out. Provide each claim as a separate sentence in a separate bullet point.

    Answer:
    {response}
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0
        )
        claims_text = completion.choices[0].message.content
    except Exception as e:
        print(f"Error in get_factual_claims: {e}")
        return []

    # Parse the claims from the output
    claims = []
    for line in claims_text.strip().split('\n'):
        line = line.strip()
        if line.startswith('- '):
            claim = line[2:].strip()
            claims.append(claim)
        elif line.startswith('* '):
            claim = line[2:].strip()
            claims.append(claim)
        elif line and (line[0].isdigit() and line[1] in ['.', ')']):
            claim = line[2:].strip()
            claims.append(claim)
    return claims

def format_dataset(queries, system_prompt):

    # Creates the responses and gets factual claims

    data = []

    for q_idx, query in enumerate(queries):

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{'role': 'system', 'content':system_prompt},
                {"role": "user", "content": query}],
            temperature=0,
            max_tokens=500
        )
        llm_response = response.choices[0].message.content.strip()
                
        sentences_ground_truth = [] # Don't really need this

        factual_claims = get_factual_claims(llm_response)
        factual_claims_truth = []

        # Loop through claims and assign them True or an Error
        for claim in factual_claims:
            
            # For code compatability
            factual_claims_truth.append(True)

        data.append([q_idx, query, llm_response, sentences_ground_truth, factual_claims, factual_claims_truth])
    return data

if __name__ == '__main__':
    utils.setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False,
        help="Keep default wandb clean.")
    parser.add_argument(
        "--wait", action=argparse.BooleanOptionalAction, default=False,
        help="Step through execution with pauses.")
    parser.add_argument(
        "--intermediate_export", action=argparse.BooleanOptionalAction, default=True,
        help="Step through execution with pauses.")
    parser.add_argument(
        "--model", type=str, default='QADebertaEntailment',
        help="Set of prompts to use.")
    parser.add_argument(
        "--n_questions", type=str, default='three',
        help="Number of questions to ask per proposition.")
    parser.add_argument(
        "--n_stochastic_questions", type=int, default=2,
        help="Number of times we generate questions.")
    parser.add_argument(
        "--n_regenerate", type=int, default=3,
        help="Number of answers per question.")
    parser.add_argument(
        "--num_data", type=int, default=int(1e19),
        help="Number of datapoints to analyse.")
    parser.add_argument(
        "--entailment_type", type=str, default='lax',  # or strict
        help="Lax or strict entailment.")
    parser.add_argument(
        "--restore_from_wandb_id", type=str, default=None,
        help=(
                "Restore (or copy) parts of a previous run. Need to also set "
                "`--restore_stages` appropriately."))
    parser.add_argument(
        "--restore_stages", default=[], nargs='*', type=str,
        help=(
            "Which stages to restore. Choices = "
            "[{model.gen_qs}, {model.answer_qs}, {model.equivalence}]"))
    parser.add_argument(
        "--accept_restore_failure", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Safely recover from restore failures. Use with care, as usually "
            "restore failures indicate you might not be restoring what you "
            "think you are. An exception to this is adding more questions to a run!"))

    args = parser.parse_args()

    df = pd.read_csv("combined_scores_with_krippendorff_alpha.csv")
    system_prompt = "Answer the query from the student. Keep the response to 100 words or less."

    data = format_dataset(df['Body'], system_prompt)

    restored = defaultdict(list)
    restored_config = dict()

    kwargs = dict(
        n_questions=args.n_questions, n_regenerate=args.n_regenerate,
        n_stochastic_questions=args.n_stochastic_questions,
        restored=restored, restore_stages=args.restore_stages,
        accept_restore_failure=args.accept_restore_failure,
        entailment_type=args.entailment_type)
    model = models.all_models[args.model](**kwargs)

    log_prompts = model.get_all_prompts_for_log()

    wait = lambda: 0  # pylint: disable = unnecessary-lambda-assignment

    results = dict(
        prompts=log_prompts, restored_config=restored_config,
        questions=dict(), metrics=dict())

    all_labels, all_uncertainties = [], []
    # Iterate over dataset
    
    for datum in data:
        didx, user_question, init_reply, init_reply_labels, facts, facts_labels = datum

        propositions, labels = facts, facts_labels

        results['questions'][f'datum-{didx}'] = dict(
            user_question=user_question, init_reply=init_reply,
            init_reply_labels=init_reply_labels, facts=facts,
            facts_labels=facts_labels, uncertainties=[],
            propositions=dict())
        ru = results['questions'][f'datum-{didx}']

        # Iterate over extracted `facts` for each individual.
        for pidx, (proposition, label) in enumerate(zip(propositions, labels)):

            text_so_far = ' '.join(propositions[:pidx]) if pidx > 0 else None

            results['questions'][f'datum-{didx}']['propositions'][f'prop-{pidx}'] = {}
            # Estimate uncertainty of proposition.
            uncertainty = model.check_truth(
                rp=ru['propositions'][f'prop-{pidx}'],
                wait=wait,
                data=dict(
                    didx=didx,
                    user_question=user_question,
                    proposition=proposition,
                    text_so_far=text_so_far)
            )

            ru['uncertainties'].append(uncertainty)
            all_uncertainties.append(uncertainty)
            all_labels.append(label)

# Get question level uncertainties
question_uncertanties = []
for data_indx, datum in results['questions'].items():

    quesiton_uncertainty = np.mean(datum['uncertainties'])
    question_uncertanties.append(quesiton_uncertainty)

data = [results, all_uncertainties, question_uncertanties]
pickle.dump(data, open(f'results_{args.model}.pickle','wb'))
