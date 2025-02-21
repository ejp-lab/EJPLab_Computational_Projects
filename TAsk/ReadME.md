# TAsk: An Open-Source Pipeline for Enhanced Graduate-Level Education

TAsk is an open-source AI system designed to study a graduate-level class in biological chemistry using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Developed to assist students in an advanced biological chemistry course at the University of Pennsylvania, TAsk integrates course materials—including lecture transcripts and academic papers—to provide accurate, context-specific responses to student queries.

## General Information
The TAsk server faciliates a student to AI pipeline which can be mediated and recorded by an instructor of a course. The pipeline features: 1) Security measures to ensure student information is redacted, 2) retrieval augmented generation to find relavent documents to a user query, 3) automatic reading and sending of emails. 

## Installation

Installation of TAsk requires conda and the installation of the associated yml script. You can install conda via instructions on the [anaconda website](https://docs.anaconda.com/anaconda/install/)

Installation of the yml file can be accomplished with the following command:
`conda env create -f TAsk.yml`

The TAsk server was originally built with the Google Cloud, ngrock, and OpenAI APIs. The following instructions will instruct on how to properly configure these endpoints for utilization of TAsk.

### ngrock

1. Create a free ngrock account.
2. Download [ngrock](https://download.ngrok.com/linux) onto a linux machine
3. In a terminal, start the server with ngrok http http://localhost:8080

### Google Cloud - Web Interface

1. Create a Google account or login to a new one and navigate to [Google Cloud](https://console.cloud.google.com/welcome/new)
2. Click the top left bars of the webpage, click APIs & Services, and Enable APIs & Services
3. On the new webpage click + Enable APIs and Services
4. In the search bar, search 'gmail' and click on the Gmail API
5. Click enable. Do the same for the Pub/Sub API by searching and enabling.
6. Under Google Auth Platform, select credentials
7. Provide an app name and an email. Click save and continue
8. Click add and remove scopes
9. In the Manually add scopes field, add the following scopes: 'https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify','https://www.googleapis.com/auth/gmail.send'
12. Fill the rest of the form out and save
13. Under Credentials click + CREATE CREDENTIALS > Create OAuth client ID
14. Application Type is desktop application. Click create after giving it a name.
15. Click DOWNLOAD JSON and name the file 'credentials.json' in the TAsk folder
16. Under Pub/sub, create a new topic with default parameters
17. Under subscriptions, create a new subscription
18. Select type 'push' and in the endpoint url, copy paste the endpoint found in the terminal from ngrok (E.g. https://xxxx-xxxx-xxxx-x-xx-xxxx-xxxx-xxxx-xxxx.ngrok.free.app). You will have to update this each time the server is restarted with the free version!
19. Click update

### OpenAI

1. Create an [OpenAI API account](https://platform.openai.com/signup)
2. Create a [Secret Key](https://platform.openai.com/settings/organization/api-keys)
3. Save the secret key to ~/.bashrc as the variable OPENAI_API_KEY (e.g. `export OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxx)
4. Add credits by entering your billing information under [billing](https://platform.openai.com/settings/organization/billing/overview)

### Credentials File

A csv file with the following columns must be present if the check_creds option is selected: use_chatgpt (1 for yes, 0 for no), email_for_chatgpt

## Usage
```
usage: run_server_github.py [-h] --sender_email SENDER_EMAIL --topic_name
                            TOPIC_NAME [--record_responses]
                            [--lecture_materials_dir LECTURE_MATERIALS_DIR]
                            [--log_name LOG_NAME] [--check_creds CHECK_CREDS]

Process command-line arguments.

options:
  -h, --help            show this help message and exit
  --sender_email SENDER_EMAIL
                        String for the sender's email
  --topic_name TOPIC_NAME
                        String for the project name
  --record_responses    Record responses (default: False)
  --lecture_materials_dir LECTURE_MATERIALS_DIR
                        Path of the lecture materials directory
  --log_name LOG_NAME   Title of the log to save
  --check_creds CHECK_CREDS
                        Path to the credentials file for whitelisted senders
```

## Analysis Scripts

**analysis.py** - Contains figures made for SI. Usage:
`python Analysis.py`

**augment_dataset.py** - Asks ChatGPT-4o to reword the questions slightly and also scrambles the answers, while maintaining the correct answer. Training and test sets used in p_true_multiple_choice.py. Usage:
`python augment_dataset.py`

*contact_chatgpt.py* - Methods for accessing OpenAI models

*email_checker.py* - Methods to access Google Cloud and emails

**evaluate_RAG_models_multiple_choice.py** - Evaluates the Perplexity AI model or TAsk on the biological chemistry multiple choice test. Usage:
`python evaluate_RAG_models_multiple_choice.py [model]`
models: TAsk, Perplexity
Note: You will need to create an [account](https://www.perplexity.ai/settings/api) with Perplexity. Save the API key as PERPLEXITY_API_KEY in .bashrc

**graph_p_true_results.py** - Creates AUROC plots for models on an augmented multiple choice set. Must run the p_true_multiple_choice.py with all models. Usage:
`python graph_p_true_results_multiple_choice.py`

**graph_results_of_ptrue_semantic_entropy.py** - Creates histograms for p_true and semantic entropy. Must run uncertainty pipeline with ptrue and semantic entropy. Usage:
`python graph_results_of_ptrue_semantic_entropy.py`

**huggingface_models.py** - Classes to help with calculation of ptrue and semantic entropy

*PrivacyFilter.py* - Classes to censor student information in emails

*p_true.py* - Methods to run p_true benchmark

**p_true_multiple_choice.py** - Runs the p_true metric for various specified models. Usage:
`python p_true_multiple_choice.py [model] [set]`
Possible models: gpt-4o, gpt-4o-mini, llama-8B, llama-70B
Possible sets: train, test

*models.py* - Classes to run the semantic entropy and p_true benchmark on pargaraph level

**uncertainty_pipeline.py** - Runs the p_true metric or semantic entropy.
```
usage: python uncertainty_pipeline_github.py [-h] [--debug | --no-debug]
                                      [--wait | --no-wait]
                                      [--intermediate_export | --no-intermediate_export]
                                      [--model MODEL]
                                      [--n_questions N_QUESTIONS]
                                      [--n_stochastic_questions N_STOCHASTIC_QUESTIONS]
                                      [--n_regenerate N_REGENERATE]
                                      [--num_data NUM_DATA]
                                      [--entailment_type ENTAILMENT_TYPE]
                                      [--restore_from_wandb_id RESTORE_FROM_WANDB_ID]
                                      [--restore_stages [RESTORE_STAGES ...]]
                                      [--accept_restore_failure | --no-accept_restore_failure]
```
*utils.py* - Contains functions for p_true and semantic entropy pipeline

## Files

**permissions.csv** - Contains individual emails and access permission for TAsk

**deidentified_dataset_final.csv** - All TAsk interactions from the 9 week study

**deidentified_reviewer_scores.csv** - Contains all assesments of the 50 representative queries from the 9 week study as well as TA answers.

## License

TAsk is released under the MIT license.

## Acknowledgements

For more detailed information, please refer to our paper and the documentation within this repository.