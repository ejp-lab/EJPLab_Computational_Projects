import os
import random
import pickle
import openai
from tqdm import tqdm
import time
import pdb
from p_true import *
from base_model import *
from huggingface_models import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from vectorize_documents import find_top_k_matching_chunks
import json
from tqdm import tqdm
import pickle

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_dataset(pickle_file):
    """Load the dataset from a pickle file."""
    with open(pickle_file, "rb") as f:
        dataset = pickle.load(f)
    return dataset

def split_dataset(dataset, train_size=30):
    """Split the dataset into train and test sets."""
    random.shuffle(dataset)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

def generate_rewordings(question_text, num_rewordings=5):
    """Generate reworded versions of a question using OpenAI GPT-4."""
    rewordings = []
    attempts = 0
    max_attempts = 10
    for _ in range(num_rewordings):
        while attempts < max_attempts:
            try:

                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an assistant that rephrases multiple-choice questions without changing their meaning or the correct answer."},
                        {"role": "user", "content": f"Please rephrase the following question, ensuring the intended answer remains the same:\n\n{question_text}"}
                    ],
                    temperature=1,
                    max_tokens=500,
                )

                reworded_question = response.choices[0].message.content.strip()
                # Ensure no duplicates
                if reworded_question not in rewordings:
                    rewordings.append(reworded_question)
                break  # Exit the retry loop on success
            except Exception as e:
                print(f"Error generating rewording: {e}")
                attempts += 1
                time.sleep(5)  # Wait before retrying
        if attempts == max_attempts:
            print("Max attempts reached. Skipping this rewording.")
    return rewordings

def shuffle_choices(choices_dict, correct_letter):
    """Shuffle answer choices while keeping track of the correct answer."""
    # Convert choices_dict to a list of tuples
    choices_list = list(choices_dict.items())  # [('A', 'Choice A text'), ...]
    random.shuffle(choices_list)
    # Map old letters to texts
    letter_to_text = choices_dict
    correct_text = letter_to_text[correct_letter]
    # Assign new letters
    new_letters = ['A', 'B', 'C', 'D', 'E']
    new_choices = []
    for i, (_, text) in enumerate(choices_list):
        new_choices.append((new_letters[i], text))
    # Find the new correct letter
    new_correct_letter = None
    for letter, text in new_choices:
        if text == correct_text:
            new_correct_letter = letter
            break
    if new_correct_letter is None:
        raise ValueError("Correct answer not found in new choices")
    return new_choices, new_correct_letter

def process_questions(questions):
    """Generate reworded questions and shuffle choices."""
    expanded_set = []
    for item in tqdm(questions, desc="Processing questions"):
        question_text = item['question']
        choices = [
            ('A', item.get('Choice A', '')),
            ('B', item.get('Choice B', '')),
            ('C', item.get('Choice C', '')),
            ('D', item.get('Choice D', '')),
            ('E', item.get('Choice E', ''))
        ]
        choices = dict(choices)
        correct_answer = item['answers']['text']  # Should be a letter like 'A'
        context = item.get('context', '')
        question_number = item['question number']
        # Generate rewordings
        rewordings = generate_rewordings(question_text, num_rewordings=5)
        for reworded_question in rewordings:
            # Shuffle the choices
            new_choices, new_correct_answer = shuffle_choices(choices, correct_answer)
            # Create the full question text including choices

            full_question_text = f"{reworded_question}\nAnswer:"
            # Append to expanded set
            expanded_set.append({
                'question': full_question_text,
                'answers': {'text':new_correct_answer},
                'choices': new_choices,
                'context': context,
                'question_number': question_number,
            })
    return expanded_set

def save_dataset(dataset, filename):
    """Save the dataset to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

df = pd.read_csv('multiple_choice_questions.csv')

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
lecture_materials_dir = "lecture_materials"
embeddings_file = 'embeddings.json'

model = HuggingfaceModel('llama', max_new_tokens=1, stop_sequences='default')


if not os.path.exists("multiple_choice_with_2_chunks_context.pickle"):
    chunk_embeddings = {}
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                for chunk_text, chunk_data in data.items():
                    chunk_embeddings[chunk_text] = chunk_data

    # Initialize the dataset and probabilities list
    dataset = []

    # Loop through the DataFrame and construct the dataset
    for idx, row in tqdm(df.iterrows()):
        question = row['Question']
        answer = row['Answer']
        choices = "\n".join([
                f"A. {row['Choice A']}",
                f"B. {row['Choice B']}",
                f"C. {row['Choice C']}",
                f"D. {row['Choice D']}",
                f"E. {row['Choice E']}",
            ])
        full_question = f'{question}\n\n{choices}\nAnswer: '
        best_text = find_top_k_matching_chunks(question, chunk_embeddings, 2)

        # This needs to be done because if something is out of the tokenizer vocab
        # We have issues later when comparing the input text to the output text
        best_text = model.tokenizer.decode(model.tokenizer(best_text)['input_ids'], skip_special_tokens=True)
        full_question = model.tokenizer.decode(model.tokenizer(full_question)['input_ids'], skip_special_tokens=True)

        if best_text.startswith(". ") or best_text.startswith(", "):
            best_text = best_text[2:]

        dataset.append({'question': question, 'context': best_text, 'answers': {'text': answer},
                        'Choice A': row['Choice A'], 'Choice B': row['Choice B'], 'Choice C': row['Choice C'],
                        'Choice D': row['Choice D'], 'Choice E': row['Choice E'], 'question number':row['Question Number']})

    pickle.dump(dataset, open("multiple_choice_with_2_chunks_context.pickle",'wb'))

else:
    dataset = pickle.load(open("multiple_choice_with_2_chunks_context.pickle",'rb'))


# Split into train and test sets
train_questions, test_questions = split_dataset(dataset, train_size=30)

# Process train and test questions
expanded_train_set = process_questions(train_questions)
expanded_test_set = process_questions(test_questions)

# Verify the counts
print(f"Number of train questions: {len(expanded_train_set)}")
print(f"Number of test questions: {len(expanded_test_set)}")

# Save the expanded datasets
save_dataset(expanded_train_set, 'train_questions_expanded.pickle')
save_dataset(expanded_test_set, 'test_questions_expanded.pickle')

