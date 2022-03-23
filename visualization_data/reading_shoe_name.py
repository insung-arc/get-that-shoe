import os

print("[INFO] Read file directories...")
print("#############START##############")

shoes_dir = []
path_dir = "../data/"
shoes_count = int(len(os.listdir(path_dir)) - 1)

for x in os.listdir(path_dir):
    dir_name = os.path.basename(x)
    if(dir_name == ".DS_Store"):
        continue
    print(dir_name)

    for i in range(shoes_count):
        shoes_dir[i] = dir_name

print("#############END################")
print(type(shoes_dir))
