from email_checker import *
from contact_chatgpt import *
import os
import time
from vectorize_documents import *
import sys
import csv
import datetime
from flask import Flask, request, jsonify
import threading
import argparse

app = Flask(__name__)

# Argument parser setup
parser = argparse.ArgumentParser(description='Process command-line arguments.')
parser.add_argument('--sender_email', type=str, required=True, help='String for the sender\'s email')
parser.add_argument('--topic_name', type=str, required=True, help='String for the project name')
parser.add_argument('--record_responses', action='store_true', default=False, help='Record responses (default: False)')
parser.add_argument('--lecture_materials_dir', type=str, default='./lecture_materials', help='Path of the lecture materials directory')
parser.add_argument('--log_name', type=str, default='server_log.csv', help='Title of the log to save')
parser.add_argument('--check_creds', type=str, default='', help='Path to the credentials file for whitelisted senders')

args = parser.parse_args()

# Assign variables from parsed arguments
sender_email = args.sender_email
topic_name = args.topic_name
record_responses = args.record_responses
lecture_materials_dir = args.lecture_materials_dir
csv_file_path = args.log_name
check_creds_flag = args.check_creds

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

service = get_service()
chunk_embeddings = vectorize_documents(lecture_materials_dir)

if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Sender", "Subject", "Body", "Response", "Document", "Error"])

watching_response = watch_gmail(service, topic_name)
print(watching_response)

last_run_finished = True

# Main loop to check email
@app.route('/', methods=['POST'])
def notifications():

    global last_run_finished
    print("Received push notification.")

    if not last_run_finished:
        return jsonify(success=True), 200

    last_run_finished = False

    try:
        subjects, bodies, senders, thread_ids = get_messages(service, check_creds_flag)
    except Exception as e:
        print(f"An error occurred: {e}")
        last_run_finished = True
        return jsonify(success=True), 200

    for i, body in enumerate(bodies):

        if check_creds_flag:
            allow_response, record_response = check_creds(senders[i], check_creds_flag)
        else:
            allow_response = True
            record_response = True

        if allow_response:
            try:
                # Clean bodies of PII
                body = clean_message(body)
            except Exception as e:
                print(e)
                last_run_finished = True
                return jsonify(success=True), 200

            try:
                # RAG implementation
                best_document = find_best_matching_document(body, chunk_embeddings)
                best_document_path = os.path.join(lecture_materials_dir, best_document)
                # Asking OpenAI's assistant
                chat_response = ask_assistant(body, OPENAI_KEY, best_document_path)
            except Exception as e:
                print(f"Error in contacting ChatGPT or document: {e}")
                last_run_finished = True
                return jsonify(success=True), 200

            try:
                # Formatting, sending, and recording the email
                subject = "[DO NOT REPLY] " + subjects[i]
                sender = senders[i]
                thread_id = thread_ids[i]

                response = chat_response + f"\n\n\nThe file name this information was pulled from is: {best_document}"
                response += "\n\n\nTHIS IS A RESPONSE FROM A ROBOT.\n\n\nDO NOT REPLY."

                print(f"Subject: {subject}\n")
                print(f"Body: {body}\n")
                print(f"Response: {response}\n")

                send_responses(service, 'me', subject, body, sender, response, thread_id, sender_email)

                if record_responses and record_response:
                    try:
                        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            clean_chat_response = chat_response.replace('\n', '').replace('\r', '')
                            writer.writerow([current_time, sender, subject, body, clean_chat_response, best_document, ""])
                    except Exception as log_error:
                        print(f"Logging error: {log_error}")

            except Exception as e:
                print(f"Subject: {subject}\n")
                print(f"Body: {body}\n")
                print(f"Response: {response}\n")
                print(f"Error in main: {e}\n")
                last_run_finished = True
                continue
        else:
            # For those who do not have access or malicious actors
            response = f"You do not have access to Dr. ChatGPT. If you would like to have access, or believe this is an error contact:\n {sender_email}"
            sender = senders[i]
            thread_id = thread_ids[i]
            subject = "[DO NOT REPLY] " + subjects[i]
            send_responses(service, 'me', subject, body, sender, response, thread_id, sender_email)

    last_run_finished = True

    return jsonify(success=True), 200

app.run(port=8080)
