import os, time, json

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

def get_dir_list(dir_path):
    lst = []
    for s in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, s)):
            lst.append(s)
    return lst

def load_dict_json(filepath, ):
    with open(filepath, encoding='utf-8') as f:
        a = json.load(f)
    return a

def save_dict_json(filepath, dic):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dic, ensure_ascii=False))
