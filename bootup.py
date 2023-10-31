import os

lock_file_path = "/content/Medical-Malpractice-Analysis/.lock_file.txt"
if os.path.exists(lock_file_path):
    print('script has already been ran')
else:
    os.chdir('/content/Medical-Malpractice-Analysis/')
    open(lock_file_path,'w').close()
    print('Google Colab initalized correctly')