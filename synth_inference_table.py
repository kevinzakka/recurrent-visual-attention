import csv
import os
import random
import sys

def generate_random_float():
    return random.uniform(0, 1)

def write_to_file(rows, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

if len(sys.argv) != 3:
    print("Error: Exactly two arguments are required (input and output filenames)")
    sys.exit(1)

if not all(isinstance(arg, str) for arg in sys.argv[1:]):
    print("Error: Both arguments must be string paths")
    sys.exit(1)

input_filename = sys.argv[1]
output_filename = sys.argv[2]

if not input_filename.endswith(".csv") or not output_filename.endswith(".csv"):
    print("Error: Both filenames must have the '.csv' file extension")
    sys.exit(1)

if not os.path.isfile(input_filename):
    print("Error: Input file does not exist")
    sys.exit(1)

with open(input_filename) as f:
    reader = csv.reader(f)
    selected_rows = []
    for row in reader:
        random_float = generate_random_float()
        if random_float <= 0.1:
            selected_rows.append(row)

write_to_file(selected_rows, output_filename)
