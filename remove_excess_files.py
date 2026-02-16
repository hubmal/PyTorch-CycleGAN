import argparse
import os
import random

def delete_random_files(directory_path, num_to_delete=150):
    all_files = [f for f in os.listdir(directory_path) 
                 if os.path.isfile(os.path.join(directory_path, f))]
    if len(all_files) < num_to_delete:
        print(f"Error: Only {len(all_files)} files found. Cannot delete {num_to_delete}.")
        return
    files_to_delete = random.sample(all_files, num_to_delete)

    print(f"Starting deletion of {num_to_delete} files from: {directory_path}")
    
    count = 0
    for file_name in files_to_delete:
        file_path = os.path.join(directory_path, file_name)
        try:
            os.remove(file_path)
            count += 1
            print(f"Deleted: {file_name}")
        except Exception as e:
            print(f"Failed to delete {file_name}: {e}")

    print(f"\nSuccessfully deleted {count} files.")

parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", type=str)
parser.add_argument("--num_images", type=int)
args = parser.parse_args()
delete_random_files(args.target_dir, args.num_images)