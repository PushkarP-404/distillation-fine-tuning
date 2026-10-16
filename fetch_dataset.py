import os
import paramiko

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('127.0.0.1', port=2222, username='root', password=os.environ.get('VM_PASSWORD', 'password_123'))

sftp = ssh.open_sftp()
sftp.get('/var/ai-agent/training_data/dataset.jsonl', 'dataset.jsonl')
sftp.close()
ssh.close()

print("Dataset fetched successfully!")