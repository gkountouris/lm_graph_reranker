import os
import time

folder_path = '/path/to/your/folder'
file_limit = 100
current_idx = 0

while True:  # Infinite loop
    # Count the number of files in the folder
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    # If less than 100 files, create a new file
    if num_files < file_limit:
        new_file_path = os.path.join(folder_path, f'file_{current_idx}.txt')
        with open(new_file_path, 'w') as file:
            file.write(f'Content for file {current_idx}')
        current_idx += 1

    # Sleep for a short duration to avoid overwhelming the file system
    time.sleep(1)