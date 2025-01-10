from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os.path
import base64
import email
from email.mime.text import MIMEText
import base64
from PrivacyFilter import PrivacyFilter
import pandas as pd
from bs4 import BeautifulSoup
import html

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/gmail.send']


def create_message(sender, to, subject, message_text, thread_id=None):
    """Create a message for an email."""
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    message_obj = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
    if thread_id:
        message_obj['threadId'] = thread_id
    return message_obj

def send_message(service, user_id, message):
    """Send an email message and mark it as read."""
    try:
        # Send the email
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        sent_message_id = sent_message['id']
        print(f"Message Id: {sent_message_id}")

        # Mark the sent message as read to prevent feedback loop
        service.users().messages().modify(
            userId=user_id, 
            id=sent_message_id, 
            body={'removeLabelIds': ['UNREAD']}
        ).execute()

        return sent_message

    except Exception as error:
        print(f'An error occurred: {error}')
        return None


def send_responses(service, user_id, subject, body, sender, response, thread_id, sender_email):

    receiver_email = sender

    # Create email
    message = create_message(sender_email, receiver_email, f"Re: {subject}", response, thread_id)
    
    # Send email
    send_message(service, user_id, message)


def get_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service

def check_creds(sender, creds_path):
    
    try:
        student_creds = pd.read_csv(creds_path)
        student_creds = student_creds.set_index('Email Address')
    except Exception as e:
        print(e)

    if "<" in sender:
        email_address = sender.split("<")[1].split(">")[0]
    else:
        email_address = sender

    allow_response = True if email_address in student_creds['email_for_chatgpt'] else False
    record_response = False
    if email_address in student_creds['email_for_chatgpt']:
        if student_creds['anonymized_answer'][email_address] == 1:
            record_response = True

    return allow_response, record_response

def get_messages(service, check_creds_flag, user_id='me', ):
    """Find messages that have chatgpt in subject line. Ignore others."""
    # Call the Gmail API to fetch UNREAD messages
    results = service.users().messages().list(userId=user_id, q='is:unread').execute()
    messages = results.get('messages', [])

    subjects = []
    bodies = []
    senders = []  # List to store sender's email addresses
    thread_ids = []  # List to store thread IDs

    if not messages:

        print("No unread messages found.")
    
    else:
        for message in messages:
            
            #pdb.set_trace()
            msg = service.users().messages().get(userId=user_id, id=message['id'], format='raw').execute()
            msg_str = base64.urlsafe_b64decode(msg['raw'].encode('ASCII')).decode('utf-8')
            mime_msg = email.message_from_string(msg_str)

            subject = mime_msg["Subject"]

            # [DO NOT REPLY] prevents cyclic feedback if chatgpt is in header if testing
            if 'chatgpt' in subject.lower() and '[DO NOT REPLY]' not in subject.lower():
                
                print('ChatGPT message found, initiating response.')
                #pdb.set_trace()
                sender = mime_msg["From"]
                # assume check_creds is defined elsewhere
                allow_response, record_response = check_creds(sender, check_creds_flag)

                if not allow_response:
                    print(f"Undefined user with email {sender} was sent. Skipping")
                    continue

                senders.append(sender)
                subjects.append(subject)
                # Extract the thread ID
                thread_id = msg.get('threadId')
                thread_ids.append(thread_id)

                body = ""
                if mime_msg.is_multipart():
                    for part in mime_msg.walk():
                        ctype = part.get_content_type()
                        cdispo = str(part.get('Content-Disposition'))

                        if ctype == 'text/plain' and 'attachment' not in cdispo:
                            body = part.get_payload(decode=True).decode('utf-8')
                            break
                        elif ctype == 'text/html' and 'attachment' not in cdispo:
                            html_body = part.get_payload(decode=True).decode('utf-8')
                            soup = BeautifulSoup(html_body, "html.parser")
                            body = soup.get_text()
                            # Optionally, decode HTML entities
                            body = html.unescape(body)
                            break
                else:
                    body = mime_msg.get_payload(decode=True).decode('utf-8')

                bodies.append(body)

                #pdb.set_trace()
                # Mark the message as read
                service.users().messages().modify(userId=user_id, id=message['id'], body={'removeLabelIds': ['UNREAD']}).execute()
            else:
                print(f"Skipping message with subject: {subject}")

    return subjects, bodies, senders, thread_ids


def clean_message(message):
    """ Remove first names and last names in case students leave them in email"""
    pfilter = PrivacyFilter()

    fields = {
        os.path.join('datasets', 'firstnames.csv'): {"replacement": "<NAAM>",
                                                        "punctuation": None},
        os.path.join('datasets', 'lastnames.csv'): {"replacement": "<NAAM>",
                                                    "punctuation": None},
    }  

    pfilter.initialize(clean_accents=True, nlp_filter=False, wordlist_filter=True,
                    regular_expressions = True, fields=fields)
    
    cleaned_message = message.replace("\n",' ').replace("\r",' ')
    filtered_message = pfilter.filter(cleaned_message)

    return filtered_message

def watch_gmail(service, topic_name):
    request = {
    'labelIds': ['INBOX'],
    'topicName': topic_name,
    'labelFilterBehavior': 'INCLUDE'
    }
    response = service.users().watch(userId='me', body=request).execute()
    return response

if __name__ == "__main__":

    service = get_service()
    subjects, bodies, senders, thread_ids = get_messages(service)
