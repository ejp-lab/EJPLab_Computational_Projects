import pandas as pd
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
from openai import OpenAI
import sys
import torch

model = sys.argv[1] # llama or chatgpt

# Set your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set in the environment variables

# Define the list of NUM_CHUNKS to iterate over
NUM_CHUNKS_LIST = [0, 1, 2, 3]

# Load the CSV file
csv_file_path = 'multiple_choice_questions.csv'
df = pd.read_csv(csv_file_path)

# Load the model and tokenizer
if model == 'llama':
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

elif model == 'chatgpt':
    pass

else:
    raise Exception("Argument 1 should be either llama or chatgpt")

def ask_gpt(message, openai_api_key):
    
    client = OpenAI(api_key=openai_api_key)
    try:
        for i in range(5): # 5 attempts before crashing
            
            response = client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            temperature=0,
            max_tokens=20)
            content = response.choices[0].message.content.strip()

            if content:
                return content
    except:
        return "F"
# Define different instruction templates

prompt_templates = [
    # Template 1: Original prompt (Baseline)
    "Answer the multiple choice question provided above. PROVIDE ONLY A SINGLE CAPITAL LETTER AFTER \"Answer\". The information provided within [CONTEXT] may or may not help you answer the question.",

    # Template 2: Explicit answer format
    "Please read the question and select the best answer from the options provided. Respond with only the capital letter (A, B, C, D, or E) that corresponds to your choice. Do not include any additional text.",

    # Template 3: Provide an example
    "Answer the following multiple-choice question by selecting the option that best answers it. Provide your answer as a single capital letter. For example:\nQuestion: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nAnswer: B",

    # Template 4: Reiterate the question
    "Carefully read the question and choose the most appropriate answer from the options. Provide only the capital letter corresponding to your choice.",

    # Template 5: Encourage reasoning but limit output
    "Think about the question and the context provided. After reasoning, provide your final answer as a single capital letter corresponding to the best choice. Do not include your reasoning in the answer.",

    # Template 6: Instruction to ignore unhelpful context
    "Use the provided context if it helps answer the question. If not, rely on your own knowledge. Provide your answer as a single capital letter (A, B, C, D, or E) without any extra text.",

    # Template 7: Emphasize correctness and conciseness
    "Select the correct answer from the options below. Ensure your answer is accurate. Provide only the capital letter corresponding to your choice.",
]

# Initialize a dictionary to store generated answers for each prompt template and number of chunks
generated_answers_dict = {}
for k in NUM_CHUNKS_LIST:
    for i in range(len(prompt_templates)):
        key = f"Prompt {i+1} Chunks {k}"
        generated_answers_dict[key] = []

# Loop through each number of chunks
for NUM_CHUNKS in NUM_CHUNKS_LIST:
    print(f"Processing with NUM_CHUNKS = {NUM_CHUNKS}")
    # Loop through each prompt template
    for i, instruction in enumerate(prompt_templates):
        print(f"Using Prompt Template {i+1}")
        for index, row in df.iterrows():
            question = row['Question']
            choices = "\n".join([
                f"A. {row['Choice A']}",
                f"B. {row['Choice B']}",
                f"C. {row['Choice C']}",
                f"D. {row['Choice D']}",
                f"E. {row['Choice E']}",
            ])
            correct_answer = row['Answer'].strip()

            # Get the best matching chunks based on NUM_CHUNKS
            if NUM_CHUNKS == 0:
                context = ""
            else:

                best_texts = []
                for chunk_num in range(NUM_CHUNKS):
                    best_texts.append(row[f'Resource {chunk_num}'])
                
                best_text = '\n\n'.join(best_texts)

                context = f"[CONTEXT]\n{best_text}\n[CONTEXT]"

            # Construct the messages for OpenAI API
            message = f"{question}\n\n{choices}\n\n{context}\n\n{instruction}\n\nAnswer: "
            
            if model == 'chatgpt':
                generated_answer = ask_gpt(message, openai_api_key)

            else:
                init_prompt = [{'role':'user','content':f"{question}\n\n{choices}\n\n{context}\n\n{instruction}\n\nAnswer: "}]
                prompt = pipe.tokenizer.apply_chat_template(init_prompt, tokenize=False, add_generation_prompt=True)

                # Generate the answer using the model
                output = pipe(prompt, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                generated_answer = output[0]['generated_text'][len(prompt):].strip()
            
            # Append the generated answer to the corresponding column
            key = f"Prompt {i+1} Chunks {NUM_CHUNKS}"
            generated_answers_dict[key].append(generated_answer)

# Create a DataFrame from the generated answers
grading_df = pd.DataFrame(generated_answers_dict)
grading_df['Correct Answer'] = df['Answer']

# Function to extract the answer using regex
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

# Apply the function to all 'Prompt' columns
for col in generated_answers_dict.keys():
    grading_df[col] = grading_df[col].apply(extract_answer)

# Function to grade the answers for each prompt template and number of chunks
def grade_prompts(row):
    grades = {}
    correct_answer = row['Correct Answer'].strip().upper()
    for col in generated_answers_dict.keys():
        generated_answer = row[col]
        if generated_answer:
            generated_answer = generated_answer.strip().upper()
            grade = int(generated_answer == correct_answer)
        else:
            grade = 0  # Assign 0 if no valid answer was extracted
        grades[f"Grade {col}"] = grade
    return pd.Series(grades)

# Applying the grading function and creating new columns
grading_results = grading_df.apply(grade_prompts, axis=1)
grading_df = pd.concat([grading_df, grading_results], axis=1)

# Save the results to a CSV file
grading_df.to_csv("grades_gpt4o_with_different_prompts_and_chunks.csv", index=False)

# Calculate and print the accuracy for each prompt template and number of chunks
for k in NUM_CHUNKS_LIST:
    for i in range(len(prompt_templates)):
        key = f"Prompt {i+1} Chunks {k}"
        grade_key = f"Grade {key}"
        accuracy = grading_df[grade_key].mean() * 100
        print(f"Accuracy for {key}: {accuracy:.2f}%")
