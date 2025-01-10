from openai import OpenAI
import json
import os
import time
import pdb
from PyPDF2 import PdfReader

def ask_gpt(message, openai_api_key):
    """Generic ask chatgpt"""
    try:
        for i in range(5): # 5 attempts before crashing
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": message}])
            content = response.choices[0].message.content.strip()

            if len(content.split()) == 1:
                return content
        
        return "No file found."
    except Exception as e:
        print(f"Error in contacting GPT: {e}")
        return None

def ask_assistant(initial_message, openai_api_key, file_name, block_of_text=False, text=''):

    # Makes a dummy file if the file_name is a block of text
    # Not used in ChemTAsk, but for evaluation using text sections
    if block_of_text:
        with open(file_name,'w') as f:
            f.write(text)

    client = OpenAI(api_key=openai_api_key)
    try:
        file = client.files.create(
            file=open(file_name, 'rb'),
            purpose='assistants'
        )
    except Exception as e:
        print(f"Error in uploading file to OpenAI: {e}")

        
    vector_store = client.beta.vector_stores.create(name="Documents")
    

    thread = client.beta.threads.create()

    assistant = client.beta.assistants.create(
    name="PennChem5520 Bot",

    instructions="""You are a helpful biochemistry tutor. You will
                    use the document to answer the question
                    the stuent asks using your retrieval tool. If the
                    answer is not in your provided document, answer
                    the question to the best of your knowledge.""",

    tools=[{"type" : "file_search"}],
    model="gpt-4-turbo-preview",
    )   
    
    file_paths = [file_name]
    file_streams = [open(path, "rb") for path in file_paths]
    
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
    )
    assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    
    message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=initial_message
    )

    if block_of_text:
        os.remove(file_name)

    for i in range(5):
        run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
        )

        while run.status != 'completed':
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            if run.status == 'failed':
                print("Message to ChatGPT Failed...")
                break
        
        if run.status == 'failed':
            continue

        
        all_messages = client.beta.threads.messages.list(
        thread_id=thread.id
        )
        
        # For loop added because sometimes the model will give empty response
        # if the response is too long (maybe?).
        final_message = all_messages.model_dump()['data'][0]['content'][0]['text']['value']
        if final_message == initial_message:
            print("Response Failed")
            continue
        else:


            print("Message Success!")

            return final_message
    
    
    print("ChatGPT failed after recursive call!")
    return "The document fetched was likely too long and the assistant forgot to respond."

def get_document(document):
    
    try:
        with open(document, 'r') as f:
            document_text = f.readlines()
        
        return json.dumps({"document": document, "text": document_text})

    except Exception as e:
        error = f"Error in reading document: {e}"
        print(error)
        return error

def get_doc_paths(dir):

    available_folders =  [os.path.join(dir,folder) for folder in os.listdir(dir)]
    available_documents = []
    for folder in available_folders:
        documents = [os.path.join(folder, document) for document in os.listdir(folder)]
        [available_documents.append(document) for document in documents]
    
    return available_documents

if __name__ == "__main__":

    OPENAPI_KEY = os.getenv("OPENAI_API_KEY")
    documents = get_doc_paths("lecture_materials")
    file_name = ask_gpt(f"Return only the file path and no other text that will answer the following message: What is DNA? File names: {documents}", OPENAPI_KEY)
    response = ask_assistant("What is DNA from lecture 1?", OPENAPI_KEY, file_name)
    print(response)
