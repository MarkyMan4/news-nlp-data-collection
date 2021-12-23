from remote import RemoteConnection
import json

# load the secrets for Reddit and the database
with open('secrets.json') as file:
    secrets = json.load(file)

    server = secrets['compute_server']
    compute_username = secrets['compute_username']
    compute_password = secrets['compute_password']

rc = RemoteConnection(server, compute_username, compute_password)
print(rc.execute_command('ls'))