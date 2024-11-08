import os
import paramiko
import stat

def download_directory(sftp, remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = os.path.join(remote_dir, entry.filename)
        local_path = os.path.join(local_dir, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            download_directory(sftp, remote_path, local_path)
        else:
            sftp.get(remote_path, local_path)

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('10.198.138.232', username='cgac', password='gac81-344', port=2222)

    sftp = ssh.open_sftp()
    remote_dir = '/Public/gct_backup/xecepton/data'
    local_dir = '/home/cgac/siw_40/data/'
    
    download_directory(sftp, remote_dir, local_dir)
    
    sftp.close()
    ssh.close()

if __name__ == "__main__":
    main()