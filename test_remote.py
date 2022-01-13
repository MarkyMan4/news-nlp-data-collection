from remote import RemoteConnection
import json

# load the secrets for Reddit and the database
with open('secrets.json') as file:
    secrets = json.load(file)

    server = secrets['compute_server']
    compute_username = secrets['compute_username']
    compute_password = secrets['compute_password']
    remote_scp_path = secrets['remote_scp_path']

input_file = 'test.txt'
rc = RemoteConnection(server, compute_username, compute_password)
rc.copy_file_to_server(input_file, f'{remote_scp_path}{input_file}')
rc.close_connection()