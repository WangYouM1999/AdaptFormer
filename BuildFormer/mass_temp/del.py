import os

directory = '/home/wym/projects/BuildFormer/fig_results/massbuilding'

# 遍历目录下的所有文件
# 遍历目录下的所有文件和子目录
# for root, dirs, files in os.walk(directory):
#     for filename in files:
#         if filename.startswith('new'):
#             file_path = os.path.join(root, filename)
#             os.remove(file_path)
#             print(f"Deleted: {file_path}")

def delete_files_with_prefix(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs[:]:
            if dir_name.startswith(prefix):
                dir_path = os.path.join(root, dir_name)
                print(f"Deleting directory: {dir_path}")
                for file_name in os.listdir(dir_path):
                    if file_name.startswith(prefix):
                        file_path = os.path.join(dir_path, file_name)
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                os.rmdir(dir_path)

delete_files_with_prefix(directory, 'new')