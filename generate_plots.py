import os
import pandas as pd
import argparse

import pandas as pd
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Process data from a CSV file')
parser.add_argument('file', type=str, help='path to the CSV file')

args = parser.parse_args()

# Get the file path from the arguments
file_path = args.file

font = {'family' : 'serif','size' : 10}


def parse_csv():
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
    x_quant_0 = df.iloc[36:54:3]
    x_quant_0 = x_quant_0.append(df.loc[0])
    x_quant_0 = x_quant_0.sort_values('size_ht')

    x_quant_1 = df.iloc[37:54:3]
    x_quant_1 = x_quant_1.append(df.loc[1])
    x_quant_1 = x_quant_1.sort_values('size_ht')

    x_quant_2 = df.iloc[38:54:3]
    x_quant_2 = x_quant_2.append(df.loc[2])
    x_quant_2 = x_quant_2.sort_values('size_ht')

    # plot the line chart
    plt.plot(x_quant_0['size_ht'], x_quant_0['acc'], c='b', label = 'ht not quantized', linewidth = 1, linestyle='-', marker='.')
    plt.plot(x_quant_1['size_ht'], x_quant_1['acc'], c='r', label = 'ht quantized 1', linewidth = 1, linestyle='-', marker='.')
    plt.plot(x_quant_2['size_ht'], x_quant_2['acc'], c='g', label = 'ht 1uantized 2', linewidth = 1, linestyle='-', marker='.')

    plt.xlabel('size ht')
    x_ticks = list(range(16,129,16))
    plt.xticks(x_ticks)
    plt.ylabel('Accuracy [%]')
    plt.ylim(0.65, 1.00)
    plt.yticks([5*i/100 + 0.70 for i in range(6)])
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=10)

    plt.title('Comparison of accuracy for different size_ht values')


    # SAVE GRAPH IN PDF
    filename = os.path.basename(file_path) # Extract filename
    root, ext = os.path.splitext(filename)  # split filename and extension
    parts = root.split('-')    # Split filename into components

    parts[0] = "graphs"    # Modify the desired part
    dir_name = '-'.join(parts[:])   # Join the parts back into a string with the directory name

    parts[0] = "graph"    # Modify the desired part
    graph_name = '-'.join(parts[:]) + '.pdf'  # create new filename with new extension

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(os.path.join(dir_name, graph_name), bbox_inches='tight',dpi=300)

    return plt

df = parse_csv()

plt = generate_plot_size_ht(df)

plt.show()