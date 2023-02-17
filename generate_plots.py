import pandas as pd
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Define a dictionary to map the old headers to the new headers
    header_map = {
        '# P': 'num_patches',
        '# G': 'num_glimpses',
        'P S': 'patch_size',
        'G S': 'glimpse_scale',
        'size': 'size_ht',
        '# Ep': 'num_epochs',
        'B E': 'best_epoch',
        'B E Ac': 'best_ep_acc',
        'Ac': 'acc',
        'Er': 'err',
        'gt': 'gt_train',
        'ht': 'ht_train',
        'phi': 'phi_train',
        'l_out': 'l_out_train',
        'gt.1': 'gt_test',
        'ht.1': 'ht_test',
        'phi.1': 'phi_test',
        'l_out.1': 'l_out_test'
    }

    # Remove empty rows
    df = df.dropna(how='all')

    # Rename the headers using the dictionary
    df = df.rename(columns=header_map)

    # Reset the rows index
    df = df.reset_index(drop=True)

    # Loop through the dataframe and replace percentage strings with numbers
    for col in df:
        if "%" in str(df[col].iloc[0]):
            df[col] = df[col].str.rstrip("%").astype(float) / 100

    # Display the resulting dataframe
    pd.set_option('display.max_rows', None)
    print(df)

    return df

def generate_plot_size_ht(df):
    # filter the rows that you want to compare
    rows_to_compare = df.iloc[37:54:3]
    rows_to_compare = rows_to_compare.append(df.loc[1])
    rows_to_compare = rows_to_compare.sort_values('size_ht')

    # plot the line chart
    plt.plot(rows_to_compare['size_ht'], rows_to_compare['acc'], c='b', linewidth = 1,label = 'iteration=10', linestyle='-', marker='.')
    plt.xlabel('Size_ht')
    plt.ylabel('Accuracy')
    plt.title('Comparison of accuracy for different size_ht values')

    plt.xticks([16, 32, 48, 64, 80, 96, 128])

    return plt



parser = argparse.ArgumentParser(description='Process data from a CSV file')
parser.add_argument('file', type=str, help='path to the CSV file')

args = parser.parse_args()

# Get the file path from the arguments
file_path = args.file

df = parse_csv(file_path)

plt = generate_plot_size_ht(df)

plt.show()