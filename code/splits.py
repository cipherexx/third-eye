import os
import shutil
from sklearn.model_selection import train_test_split

input_path = '../ff_raw'
output_path = '../ff++'

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1


def clear_output_folder(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  
    os.makedirs(output_path, exist_ok=True) 


def copy_folder(src_folder, dest_subfolder):
    if os.path.isdir(src_folder): 
        shutil.copytree(src_folder, dest_subfolder)


clear_output_folder(output_path)


def split_folders(class_folder, subfolder_name):
    full_subfolder_path = os.path.join(input_path, class_folder, subfolder_name)
    
    if os.path.isdir(full_subfolder_path):
        # List all folders (videos or folders like '000_frames') inside the class folder
        folders = [item for item in os.listdir(full_subfolder_path) if os.path.isdir(os.path.join(full_subfolder_path, item))]

        train_folders, temp_folders = train_test_split(folders, test_size=1 - train_ratio, random_state=None)
        val_folders, test_folders = train_test_split(temp_folders, test_size=test_ratio / (test_ratio + val_ratio), random_state=None)

        for folder in train_folders:
            src_folder_path = os.path.join(full_subfolder_path, folder)
            dest_folder_path = os.path.join(output_path, 'train', class_folder, subfolder_name, folder)
            copy_folder(src_folder_path, dest_folder_path)
        
        for folder in val_folders:
            src_folder_path = os.path.join(full_subfolder_path, folder)
            dest_folder_path = os.path.join(output_path, 'valid', class_folder, subfolder_name, folder)
            copy_folder(src_folder_path, dest_folder_path)
        
        for folder in test_folders:
            src_folder_path = os.path.join(full_subfolder_path, folder)
            dest_folder_path = os.path.join(output_path, 'test', class_folder, subfolder_name, folder)
            copy_folder(src_folder_path, dest_folder_path)


def split_data():
    
    split_folders('real', '')
    split_folders('fake', 'Deepfakes')
    split_folders('fake', 'Face2Face')
    print("Data successfully split and copied!")

if __name__ == "__main__":
    split_data()
