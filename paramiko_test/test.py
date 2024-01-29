import paramiko
from scp import SCPClient
import os


def create_ssh_client(hostname, port, username, pem_file_path):
    """Create and return an SSH client, connected to the host."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port, username, key_filename=pem_file_path)
    return ssh

def progress(filename, size, sent):
    """ Function to track the progress of the SCP transfer """
    print(f"Transferring {filename}: {float(sent)/float(size)*100:.2f}% complete")

def scp_recursive_copy(ssh_client, local_path, remote_path, put=True):
    """Copy a directory recursively from the remote host to local host using SCP."""
    with SCPClient(ssh_client.get_transport(), progress=progress) as scp:
        if put:
            scp.put(local_path, remote_path, recursive=True)
        else:
            scp.get(remote_path, local_path, recursive=True)

class SecureCopyProtocol:
    def __init__(self, hostname, port, username, pem_file_path):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.pem_file_path = pem_file_path

        self.ssh_client: paramiko.SSHClient = self._create_ssh_client()

    def put(self, local_path, remote_path):
        """Copy a directory recursively from the remote host to local host using SCP."""
        transport = self.ssh_client.get_transport()
        assert isinstance(transport, paramiko.Transport)
        with SCPClient(transport, progress=self._progress) as scp:
            scp.put(local_path, remote_path, recursive=True)

    def get(self, remote_path, local_path):
        """Copy a directory recursively from the remote host to local host using SCP."""
        transport = self.ssh_client.get_transport()
        assert isinstance(transport, paramiko.Transport)
        with SCPClient(transport, progress=self._progress) as scp:
            scp.get(remote_path, local_path, recursive=True)

    def _create_ssh_client(self):
        """Create and return an SSH client, connected to the host."""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            self.hostname, 
            self.port, 
            self.username,
            key_filename=self.pem_file_path
        )
        return ssh
    
    @staticmethod
    def _progress(filename, size, sent):
        """ Function to track the progress of the SCP transfer """
        print(f"Transferring {filename}: {float(sent)/float(size)*100:.2f}% complete")


# Putting files
hostname = '54.176.152.39'
port = 22
username = 'nick'
pem_file_path = os.environ["USWEST1"]
remote_path = '/nvme1n1users/nick/Tmp/test'
local_path = './move'

ssh_client = create_ssh_client(hostname, port, username, pem_file_path)
scp_recursive_copy(ssh_client, local_path, remote_path, put=True)


# retreving files
hostname = '54.176.152.39'
port = 22
username = 'nick'
pem_file_path = os.environ["USWEST1"]
remote_path = '/nvme1n1users/nick/Tmp/test/move'
local_path = './receive'

ssh_client = create_ssh_client(hostname, port, username, pem_file_path)
scp_recursive_copy(ssh_client, local_path, remote_path, put=False)


scp = SecureCopyProtocol(hostname, port, username, pem_file_path)

local_path = "./move"
scp.put(local_path, remote_path)

local_path = "./receive"
scp.get(remote_path, local_path)



