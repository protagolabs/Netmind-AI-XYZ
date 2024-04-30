""" 
============
Upload Agent
============
@file_name: upload_agent.py
@author: BlackSheep Team, Netmind.AI
@description: This script is used to upload the files which can help to make docker 
image to the docker registry.
"""


import os
import ast
import json
import zipfile
import requests
import argparse
import warnings

from tqdm import tqdm


def set_args():
    
    parser = argparse.ArgumentParser(description='Upload the files to the docker registry.')
    
    parser.add_argument('--folder_path', '-fp', type=str, help='The path of the folder to upload.')
    parser.add_argument('--file', '-f', type=str, help='The name of the file which contain the api.')
    parser.add_argument('--function_name', '-fn', type=str, help='The name of the function \
                        which you want to create the api.')
    
    args = parser.parse_args()
    
    return args

def add_parent_to_nodes(node):
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parent_to_nodes(child)

def analysis_code(folder_path):

    python_files = {}
    for root, _, file_list in os.walk(folder_path):
        for file in file_list:
            if file == "requirements.txt":
                requirments_path = os.path.join(root, file)
            if file.endswith('.py'):
                python_files[file] = os.path.join(root, file)

    # 存储所有类和函数的信息
    parsed_info = {}

    # 遍历所有 Python 文件
    for file_name, file_path in python_files.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            # 解析文件并生成抽象语法树
            tree = ast.parse(f.read(), filename=file_path)
            add_parent_to_nodes(tree)

            classes = {"global": {"functions": []}}

            # 遍历抽象语法树中的节点
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # 记录类的名称和函数
                    classes[node.name] = {
                        "functions": []
                    }
                    
                elif isinstance(node, ast.FunctionDef):
                    # 记录函数的名称、参数、参数类型注解和文档字符串
                    function_info = {
                        "file_name": file_name,
                        "name": node.name,
                        "args": [(arg.arg, ast.unparse(arg.annotation) if arg.annotation else None) for arg in node.args.args],
                        "docstring": ast.get_docstring(node)
                    }
                    # 将函数信息添加到对应的类信息中
                    if node.parent and isinstance(node.parent, ast.ClassDef):
                        classes[node.parent.name]["functions"].append(function_info)
                    else:
                        classes["global"]["functions"].append(function_info)

            parsed_info[file_path] = classes
            
    parsed_info["requirements.txt"] = requirments_path

    return parsed_info

def check_related_info(parsed_info, args):
    
    file_path = ""
    exist = False
    for local_file_path, file_info in parsed_info.items():
        if args.file in local_file_path:
            file_path = local_file_path
            exist = True
    if not exist:
        raise ValueError(f"The file {args.folder_path} is not in the folder.")
            
    
    assert 'requirements.txt' in parsed_info, "The requirements.txt is not in the folder."
    
    file_info = parsed_info[file_path]
    
    functions_name = {function_info['name']: function_info for class_info in file_info.values() for function_info in class_info['functions']}
        
    assert args.function_name in functions_name, f"The function {args.function_name} is not in the file {args.file}."
    
    # TODO: 一个文件中有两个相同的函数名，报错。

    if not functions_name[args.function_name]["docstring"]:
        warnings.warn(f"Warning: The function {args.function_name} in the file {args.file} does not have docstring.")

    interface = functions_name[args.function_name]
    interface['file_path'] = file_path
    
    return interface
    
def test_program():
    # TODO: 检查程序是否是可以跑通的
    
    return True
    
def zip_files_and_folder(folder_path, extra_file_paths):
    
    if "/" in folder_path:
        zip_file_path = folder_path.split("/")[-1] + ".zip"
    else: 
        zip_file_path = folder_path + ".zip"
    
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # 遍历文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 获取文件的完整路径
                full_file_path = os.path.join(root, file)
                # 获取文件在 zip 文件中的路径
                arcname = os.path.relpath(full_file_path, folder_path)
                # 将文件添加到 zip 文件中
                zipf.write(full_file_path, arcname=arcname)

        # 遍历额外的文件路径列表
        for file_path in extra_file_paths:
            # 获取文件在 zip 文件中的路径
            arcname = os.path.relpath(file_path)
            # 将文件添加到 zip 文件中
            zipf.write(file_path, arcname=arcname)
            
    return zip_file_path
            
def zip_files_and_related(interface_info, analysis_info, folder_path):
    
    with open("interface_info.json", "w") as f:
        json.dump(interface_info, f)
    with open("analysis_info.json", "w") as f:
        json.dump(analysis_info, f) 
        
    zip_file_path = zip_files_and_folder(folder_path, ["interface_info.json", "analysis_info.json"]) 
    
    # 删除两个 json 文件
    os.remove("interface_info.json")
    os.remove("analysis_info.json")
    
    return zip_file_path

# 定义上传文件的函数
def upload_file(file_path, url, client_id, user_id):
    
    # 获取文件大小
    file_size = os.path.getsize(file_path)

    # 打开文件
    with open(file_path, 'rb') as f:
        # 创建进度条
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_path) as pbar:
            # 读取整个文件
            file_data = f.read()

            # 更新进度条
            pbar.update(file_size)

            # 发送文件
            response = requests.post(url, files={'file': (os.path.basename(file_path), file_data)}, 
                                     data={'client_id': client_id, 'user_id': user_id})

    # 返回响应
    return response


if __name__ == "__main__":

    args = set_args()
    
    parsed_info = analysis_code(args.folder_path)
    # print(parsed_info)
    
    interface = check_related_info(parsed_info, args)
    
    zip_file_path = zip_files_and_related(interface, parsed_info, args.folder_path)
    
    response = upload_file(zip_file_path, 'http://127.0.0.1:8898/upload/123/binliang', 123, "binliang")
    
    print(response.json()['message'])
