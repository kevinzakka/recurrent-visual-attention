import csv
import os
import random
import sys

def generate_random_float():
    return random.uniform(0, 1)

def write_to_file(rows, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if len(sys.argv) != 4:
    print("Error: Exactly three arguments are required (input, output filenames and percentage)")
    sys.exit(1)

input_filename, output_filename, percentage = sys.argv[1:4]

if not all(isinstance(arg, str) for arg in [input_filename, output_filename]):
    print("Error: Both filenames must be string paths")
    sys.exit(1)

if not input_filename.endswith(".csv") or not output_filename.endswith(".csv"):
    print("Error: Both filenames must have the '.csv' file extension")
    sys.exit(1)

if not os.path.isfile(input_filename):
    print("Error: Input file does not exist")
    sys.exit(1)

try:
    copy_percentage = float(percentage)
except ValueError:
    print("Error: copy_percentage must be a floating point number")
    sys.exit(1)

if not 0 <= copy_percentage <= 1:
    print("Error: copy_percentage must be between 0 and 1")
    sys.exit(1)

with open(input_filename) as f:
    reader = csv.reader(f)
    selected_rows = []
    for row in reader:
        if generate_random_float() <= copy_percentage:
            selected_rows.append(row)

write_to_file(selected_rows, output_filename)
