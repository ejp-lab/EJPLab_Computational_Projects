import os
import json
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pdb
import multiprocessing as mp

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Function to read and chunk documents
def read_and_chunk_documents(directory, chunk_size=500):
    
    documents = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        if filename.endswith(".txt") or filename.endswith(".vtt"):
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()

        elif filename.endswith(".pdf"):
            text = ""
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

        else:
            continue  # Skip non-txt and non-pdf files

        # Chunking the document
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        documents[filename] = chunks

    return documents

# Function to embed texts and optionally append to file

def clean_and_embed(text, model="text-embedding-3-large"):

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return embedding, text
    
def embed_texts(documents, model="text-embedding-3-large", embeddings_file="embeddings.json", save_to_file=True):
    
    embeddings = {}
    for doc_name, chunks in documents.items():

        for chunk in chunks:
            
            embedding, cleaned_text = clean_and_embed(chunk)
            embeddings[cleaned_text] = {'embedding': embedding, 'document': doc_name}
            
            if save_to_file:
                with open(embeddings_file, 'a') as file:
                    json.dump({cleaned_text: {'embedding': embedding, 'document': doc_name}}, file)
                    file.write('\n')
    
    return embeddings

# Updated function to find the best matching document
def find_best_matching_document(user_message, chunk_embeddings):
    
    user_embedding = embed_texts({'user_message': [user_message]}, save_to_file=False)[user_message]['embedding']
    max_similarity = -1
    best_match = None

    for chunk_text, data in chunk_embeddings.items():

        chunk_embedding = data['embedding']
        similarity = cosine_similarity([user_embedding], [chunk_embedding])[0][0]
      
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = data['document']

    return best_match

# Returns the chunk instead of the whole document
def find_best_matching_chunk(user_message, chunk_embeddings):
    
    user_embedding = embed_texts({'user_message': [user_message]}, save_to_file=False)[user_message]['embedding']
    max_similarity = -1
    best_match = None

    for chunk_text, data in chunk_embeddings.items():

        chunk_embedding = data['embedding']
        similarity = cosine_similarity([user_embedding], [chunk_embedding])[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = chunk_text

    return best_match

# Calculates cosine similarity between two texts
def calculate_similarity(user_embedding, chunk_data):
    chunk_text, chunk_embedding = chunk_data
    similarity = cosine_similarity([user_embedding], [chunk_embedding])[0][0]
    return (chunk_text, similarity)

# Returns the best matching chunks up to a specified k value
def find_top_k_matching_chunks(user_message, chunk_embeddings, k):
    user_embedding = embed_texts({'user_message': [user_message]}, save_to_file=False)[user_message]['embedding']
    
    # Create a pool of processes
    with mp.Pool(mp.cpu_count() - 4) as pool:
        # Prepare the data for parallel processing
        chunk_data = [(chunk_text, data['embedding']) for chunk_text, data in chunk_embeddings.items()]
        
        # Parallel computation of cosine similarity
        similarities = pool.starmap(calculate_similarity, [(user_embedding, data) for data in chunk_data])

    # Sort the chunks by similarity in descending order
    top_k_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    # Concatenate the top k matching chunks into a single string
    concatenated_chunks = '\n\n'.join([chunk_text for chunk_text, _ in top_k_chunks])

    return concatenated_chunks

# Opens precomputed chunk embeddings
def get_processed_files(processed_files_path):

    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as file:
            return set(file.read().splitlines())

    return set()

# Run at the begining of the server to load previously embedded files
# Creates a text file of already embedded files
# Embeds any new files found in the directory
def vectorize_documents(directory, embeddings_file="embeddings.json", processed_files_path="processed_files.txt"):
    
    # You can chunk the documents and load them from a pickle if the path already exists
    documents = read_and_chunk_documents(directory)
    processed_files = get_processed_files(processed_files_path)

    # Load existing embeddings
    chunk_embeddings = {}
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                for chunk_text, chunk_data in data.items():
                    chunk_embeddings[chunk_text] = chunk_data

    # Process new or missing documents
    for doc_name, chunks in documents.items():
        if doc_name not in processed_files:
            for chunk in chunks:
                chunk_text = chunk.replace("\n", " ")
                embedding = client.embeddings.create(input=[chunk_text], model="text-embedding-3-large").data[0].embedding
                chunk_embeddings[chunk_text] = {'embedding': embedding, 'document': doc_name}

                # Append new embeddings to file
                with open(embeddings_file, 'a') as file:
                    json.dump({chunk_text: {'embedding': embedding, 'document': doc_name}}, file)
                    file.write('\n')

            # Update the list of processed files
            with open(processed_files_path, 'a') as file:
                file.write(doc_name + '\n')

    return chunk_embeddings
