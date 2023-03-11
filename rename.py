import os

path = r'E:\python\python_project\mogushujuji\yuanshi'
index = 1
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.splitext(file_path)[-1] == '.jpg':
        new_file_path = '/'.join((os.path.splitext(file_path)[0].split('\\'))[:-1]) + '/{:0>2}.jpg'.format(index)
        index += 1
        print(file_path+'---->'+new_file_path)
        os.rename(file_path, new_file_path)
    elif os.path.splitext(file_path)[-1] == '.png':
        new_file_path = '/'.join((os.path.splitext(file_path)[0].split('\\'))[:-1]) + '/{:0>2}.jpg'.format(index)
        index += 1
        print(file_path + '---->' + new_file_path)
        os.rename(file_path, new_file_path)
