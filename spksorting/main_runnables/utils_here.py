import os

def clear_folder(_dir):
    '''Clear a folder recursively
    https://stackoverflow.com/questions/13118029/deleting-folders-in-python-recursively
    '''
    if os.path.exists(_dir):
        for the_file in os.listdir(_dir):
            file_path = os.path.join(_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    clear_folder(file_path)
                    os.rmdir(file_path)
            except Exception as e:
                print(e)