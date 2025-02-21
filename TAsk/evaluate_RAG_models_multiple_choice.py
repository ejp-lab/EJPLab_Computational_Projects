import pandas as pd
from tqdm import tqdm
import os
from openai import OpenAI
from contact_chatgpt import ask_assistant
import re
import sys

# NOTE: Evaluation of ChemTAsk in this script is not authentic
# Evaluation of ChemTAsk requires whole documents which we cannot provide
# in this Github repository due to copyright law

model = sys.argv[1] # ChemTAsk or Perplexity

def ask_perplexity(message):
    response = client.chat.completions.create(
        model='llama-3.1-sonar-huge-128k-online',
        messages=[{"role": "user", "content": message}],
        temperature=0.0,
        top_p=0.0
    )
    content = response.choices[0].message.content.strip()
    return content

NUM_CHUNKS = 1

csv_file_path = 'multiple_choice_questions.csv'
df = pd.read_csv(csv_file_path)

if model == 'ChemTAsk':
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

elif model == 'Perplexity':
    openai_api_key = os.getenv("PERPLEXITY_API_KEY")
    client = OpenAI(api_key=openai_api_key, base_url='https://api.perplexity.ai')

else:
    raise Exception('Argument 1 must be ChemTAsk or Perplexity')

# Instruction template
instruction = (
    "Answer the multiple choice question provided above. "
    "PROVIDE ONLY A SINGLE CAPITAL LETTER AFTER \"Answer\". "
)

# Initialize counters for evaluation
total_questions = len(df)
correct_answers = 0

# Loop through the dataset and evaluate the model
generated_answers_dict = {f"RAG {i+1}": [] for i in range(NUM_CHUNKS)}

# Loop through the dataset and evaluate the model
for i in range(0, NUM_CHUNKS):
    for index, row in tqdm(df.iterrows(), total=total_questions):
        question = row['Question']
        choices = "\n".join([
            f"A. {row['Choice A']}",
            f"B. {row['Choice B']}",
            f"C. {row['Choice C']}",
            f"D. {row['Choice D']}",
            f"E. {row['Choice E']}",
        ])
        correct_answer = row['Answer'].strip()
        
        best_texts = []
        for i in range(NUM_CHUNKS):
            best_texts.append(row[f'Resource {i}'])
        
        best_text = '\n\n'.join(best_texts)

        context = f"[CONTEXT]\n{best_text}\n[CONTEXT]"
        
        # Construct the prompt
        prompt = f"{question}\n\n{choices}\n\n{instruction}\n\nAnswer: "

        if model == 'ChemTAsk':
            generated_answer = ask_assistant(prompt, openai_api_key, 'dummy.txt', block_of_text=True, text=best_text)
        elif model == 'Perplexity':
            generated_answer = ask_perplexity(prompt)

        # Evaluate the generated answer
        if generated_answer == correct_answer:
            correct_answers += 1
        
        generated_answers_dict[f"RAG {i+1}"].append(generated_answer)

grading_df = pd.DataFrame(generated_answers_dict)
grading_df['Correct Answer'] = df['Answer']

def grade_rounds(row):
    grades = []
    for i in range(NUM_CHUNKS):
        grade = int(row['Correct Answer'].strip() == row[f"RAG {i+1}"].strip())
        grades.append(grade)
    return grades

def extract_answer(value):
    if not isinstance(value, str):
        return None
    # Regex pattern to match " X ", " X.", or " X" where X is A-E
    pattern = r'(?:^|\s)([A-E])(?:\s|\.|$)'
    match = re.search(pattern, value.strip())
    if match:
        return match.group(1)
    else:
        return None  # or value.strip()

# Applying the grading function and creating new columns
for col in generated_answers_dict.keys():
    grading_df[col] = grading_df[col].apply(extract_answer)

for col in generated_answers_dict.keys():
    grade_col = f"Grade {col}"
    grading_df[grade_col] = grading_df.apply(
        lambda row: int(row['Correct Answer'].strip() == (row[col] or '').strip()),
        axis=1
    )
grading_df.to_csv(f"{model}_performance.csv", index=False)

# Note you need to probably check generated_answers_dict for any missed grades
for i in range(1, NUM_CHUNKS + 1):
    grade_col = f"Grade RAG {i}"
    accuracy = grading_df[grade_col].mean() * 100
    print(f"Accuracy for {model}: {accuracy:.2f}%")



