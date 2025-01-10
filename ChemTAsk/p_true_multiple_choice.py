from p_true import *
from huggingface_models import *
import pickle
import sys

# Load the model
# gpt-4o, gpt-4o-mini, llama-8B, llama-70B
MODEL_NAME = sys.argv[1]
# train, test
SET = sys.argv[2]
model = HuggingfaceModel(MODEL_NAME, max_new_tokens=1, stop_sequences='default')

train = pickle.load(open('train_questions_expanded.pickle','rb'))
q_of_interest = train[0]['question_number']
few_shot_examples = [train.pop(i) for i, example in enumerate(train) if example['question_number'] == q_of_interest][:2]

# Formatting the few shot examples
for idx, info in enumerate(few_shot_examples):
    question = info['question'].replace("Answer:","")
    answer = info['answers']['text']
    choices = "\n".join([
            f"A. {info['choices'][0][1]}",
            f"B. {info['choices'][1][1]}",
            f"C. {info['choices'][2][1]}",
            f"D. {info['choices'][3][1]}",
            f"E. {info['choices'][4][1]}",
        ])
    full_question = f'{question}\n\n{choices}'
    few_shot_examples[idx]['question'] = full_question
    few_shot_examples[idx]['answers']['text'] = [f'{answer}',f'{answer}.',f'{answer})']
    
if SET == 'test':
    
    # Uses few shot examples from train, but replaces train in place with test
    train = pickle.load(open('test_questions_expanded.pickle','rb'))


prompt = 'Answer the following multiple-choice question by selecting the option that best answers it. Provide your answer as a single capital letter. For example:\nQuestion: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nAnswer: B'

# Loop through the dataset and calculate probabilities
collected_responses = []
answers = []
probabilities = []
correct_answers = []

# Formatting the set
for idx, info in enumerate(train):
    question = info['question'].replace("Answer:","")
    answer = info['answers']['text']
    choices = "\n".join([
            f"A. {info['choices'][0][1]}",
            f"B. {info['choices'][1][1]}",
            f"C. {info['choices'][2][1]}",
            f"D. {info['choices'][3][1]}",
            f"E. {info['choices'][4][1]}",
        ])
    full_question = f'{question}\n\n{choices}'
    
    train[idx]['question'] = full_question
    train[idx]['answers']['text'] = [f'{answer}',f'{answer}.',f'{answer})']


# Construct the few-shot prompt for later
few_shot_prompt, all_responses, it = construct_few_shot_prompt(
    model=model,
    dataset=few_shot_examples,
    indices=[i for i in range(len(few_shot_examples))],
    prompt=prompt,
    brief='',
    brief_always=False,
    make_prompt=make_prompt,
    num_generations=20,
    metric=True,
)

for i, data in enumerate(train):

    # Generate the responses for p_true
    dummy_prompt, all_responses, it = construct_few_shot_prompt(
        model=model,
        dataset=train,
        indices=[i],
        prompt=prompt,
        brief='',
        brief_always=False,
        make_prompt=make_prompt,
        num_generations=20,
        metric=None,
    )

    # Get p_true using few shot examples from earlier
    log_prob, final_prompt = calculate_p_true(
        model=model,
        question=data['question'],
        most_probable_answer=all_responses[i]['most_likely_response'],
        brainstormed_answers=all_responses[i]['responses'],
        few_shot_prompt=few_shot_prompt,
        hint=False
    )


    # Convert log probability to probability
    prob = 10 ** log_prob
    probabilities.append(prob)
    print(f'Processed {i+1}/{len(train)}: Probability = {prob}')

    collected_responses.append(all_responses)
    answers.append(all_responses[i]['most_likely_response'])
    correct_answers.append(data['answers']['text'])




grades = [1 if a in correct_answers[idx] else 0 for idx, a in enumerate(answers)]

# Separate probabilities into correct and incorrect
probs_correct = [prob for prob, grade in zip(probabilities, grades) if grade == 1]
probs_incorrect = [prob for prob, grade in zip(probabilities, grades) if grade == 0]

# Store for analysis script
data = dict(collected_responses=collected_responses,
            answers=answers,
            probabilities=probabilities,
            correct_answers=correct_answers,
            probs_correct=probs_correct,
            probs_incorrect=probs_incorrect)

pickle.dump(data, open(f'{SET}_{MODEL_NAME}_stats.pickle','wb'))