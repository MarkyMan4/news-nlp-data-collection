"""
    This is to handle SSH and SCP with another server.
"""

import json
import paramiko
from scp import SCPClient

class RemoteConnection:
    def __init__(self, server: str, username: str, password: str):
        self.server = server
        self.username = username
        self.password = password

        # create the ssh connection
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(server, username=username, password=password)

        # create the scp connection
        self.scp = SCPClient(self.ssh.get_transport())

    def execute_command(self, cmd: str) -> str:
        """
        executes a command on the server that was provided in the constructor

        Args:
            cmd (str): command to execute

        Returns:
            str: output of command if it was successful, otherwise returns 'error'
        """
        stdin, stdout, stderr = self.ssh.exec_command(cmd)

        # this is a blocking call which waits until the command finishes execution
        exit_status = stdout.channel.recv_exit_status()

        result = ''

        if exit_status == 0:
            result = stdout.read()
        else:
            result = 'error'

        return result

    def copy_file_to_server(self, local_path, remote_path):
        self.scp.put(local_path, remote_path)

    def get_file_from_server(self, remote_path):
        self.scp.get(remote_path)

    def close_connection(self):
        self.ssh.close()
        self.scp.close()


# testing - test running the 'ls' command, copying a file to the server and
#           getting a file from the server
# with open('secrets.json') as f:
#     secrets = json.load(f)

#     server = secrets['compute_server']
#     username = secrets['compute_username']
#     password = secrets['compute_password']
#     remote_path = secrets['remote_scp_path']

# con = RemoteConnection(server, username, password)

# print(con.execute_command('ls'))
# con.copy_file_to_server('test.csv', f'{remote_path}test.csv')
# con.get_file_from_server(f'{remote_path}test1.csv')

# con.close_connection()
