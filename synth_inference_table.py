import csv
import random

def generate_random_float():
    return random.uniform(0, 1)

def write_to_file(rows, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

with open('ram_5_4x4_4_3_0_0_2_0.csv') as f:
    reader = csv.reader(f)
    selected_rows = []
    for row in reader:
        random_float = generate_random_float()
        if random_float <= 0.1:
            selected_rows.append(row)

write_to_file(selected_rows, 'output.csv')
