import json

file_path = r"D:\projects\Tasks\Divine_pic_Face_Recognition\matching_Output\matches.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

def count_unique_values(json_dict, key):
    if key in json_dict:
        unique_files = set(json_dict[key])  # Convert list to set to remove duplicates
        return len(unique_files), unique_files
    else:
        return 0, set()  # Return 0 if the key is not found

key_to_check = "person_15"  # Change this to the key you want to check
unique_count, unique_files = count_unique_values(data, key_to_check)

print(f"Number of unique values in '{key_to_check}': {unique_count}")