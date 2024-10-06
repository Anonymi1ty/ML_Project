# 将Data.txt中的文本数据分为两类，分别存储到Safe和Unsafe文件夹中
import os

# Define the base path for the Data folder
data_folder = "./Data/RowData"
safe_folder = os.path.join(data_folder, "Safe")
unsafe_folder = os.path.join(data_folder, "Unsafe")

# Ensure the folders exist
os.makedirs(safe_folder, exist_ok=True)
os.makedirs(unsafe_folder, exist_ok=True)

# Initialize file counters for Safe and Unsafe outputs
safe_counter = len(os.listdir(safe_folder)) + 1
unsafe_counter = len(os.listdir(unsafe_folder)) + 1

# Read the dataset file
data_file = "./Data.txt"

try:
    with open(data_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            # Check for Safe Model Output and create files accordingly
            if "Safe Model Output:" in line:
                content = line.split("Safe Model Output:", 1)[1].strip()
                safe_file_path = os.path.join(safe_folder, f"{safe_counter}.txt")
                with open(safe_file_path, "w", encoding="utf-8") as safe_file:
                    safe_file.write(content)
                safe_counter += 1
            # Check for Unsafe Model Output and create files accordingly
            elif "Unsafe Model Output:" in line:
                content = line.split("Unsafe Model Output:", 1)[1].strip()
                unsafe_file_path = os.path.join(unsafe_folder, f"{unsafe_counter}.txt")
                with open(unsafe_file_path, "w", encoding="utf-8") as unsafe_file:
                    unsafe_file.write(content)
                unsafe_counter += 1
except FileNotFoundError:
    print(f"The file {data_file} was not found.")