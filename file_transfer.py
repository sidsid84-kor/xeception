import paramiko
import os

filelist = []
with open('list.txt') as f:
    for files in f.readlines():
        filename = files.replace('\n', '')
        filelist.append(filename)

# SFTP 연결 설정
sftp_host = '113.198.138.232'
sftp_port = 2222
sftp_username = 'cgac'
sftp_password = 'gac81-344'

# SFTP 연결 생성
transport = paramiko.Transport((sftp_host, sftp_port))
transport.connect(username=sftp_username, password=sftp_password)
sftp = paramiko.SFTPClient.from_transport(transport)


# 원본 디렉토리 및 대상 디렉토리 설정
source_base_directory = 'D:/images_backup/modified_images/'
destination_directory = '/Home/contactlensEB_backup/silgum_20240812/'

try:
    failed_count = 0
    # 파일 이동

    for row in filelist:
        try:
            # 파일명에서 날짜 추출 (정규표현식 사용)
            date_folder = row[:8]
            for rf in os.listdir(f"{source_base_directory}{date_folder}"):
                if row in rf:
                    rfilename = rf
            source_path = f"{source_base_directory}{date_folder}/{rfilename}"
            destination_path = f"{destination_directory}/{rfilename}"
            
            # 임시 로컬 디렉토리의 파일을 대상 디렉토리로 업로드
            sftp.put(source_path, destination_path)
        except Exception as e:
            failed_count += 1
            continue
except Exception as e:
    print(e)
    
finally:
    sftp.close()
    transport.close()