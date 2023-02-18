import os
import numpy as np
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

def generate_plot_num_axis(df, start_row, end_row, col, x_label, x_ticks, plt_title):
    # filter the rows that you want to compare
    x_quant_0 = df.iloc[start_row : end_row+1 : 3]
    x_quant_0 = x_quant_0.append(df.loc[0])
    x_quant_0 = x_quant_0.sort_values(col)

    x_quant_1 = df.iloc[start_row+1 : end_row+2 : 3]
    x_quant_1 = x_quant_1.append(df.loc[1])
    x_quant_1 = x_quant_1.sort_values(col)

    x_quant_2 = df.iloc[start_row+2 : end_row+3 : 3]
    x_quant_2 = x_quant_2.append(df.loc[2])
    x_quant_2 = x_quant_2.sort_values(col)

    # plot the line chart
    plt.plot(x_quant_0[col], x_quant_0['acc'], c='b', label = 'ht not quantized', linewidth = 1, linestyle='-', marker='.')
    plt.plot(x_quant_1[col], x_quant_1['acc'], c='r', label = 'ht quantized 1', linewidth = 1, linestyle='-', marker='.')
    plt.plot(x_quant_2[col], x_quant_2['acc'], c='g', label = 'ht 1uantized 2', linewidth = 1, linestyle='-', marker='.')

    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    plt.ylabel('Accuracy [%]')
    plt.ylim(0.65, 1.00)
    plt.yticks([5*i/100 + 0.70 for i in range(6)])
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=10)

    plt.title(plt_title)


    # SAVE GRAPH IN PDF
    filename = os.path.basename(file_path) # Extract filename
    root, ext = os.path.splitext(filename)  # split filename and extension
    parts = root.split('-')    # Split filename into components

    parts[0] = "graphs"    # Modify the desired part
    dir_name = '-'.join(parts[:])   # Join the parts back into a string with the directory name

    parts[0] = "graph"    # Modify the desired part
    parts.append(col)
    graph_name = '-'.join(parts[:]) + '.pdf'  # create new filename with new extension

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(os.path.join(dir_name, graph_name), bbox_inches='tight',dpi=300)

    return plt

def generate_plot_string_axis(df, start_row, end_row, col, x_label, plt_title):
    # filter the rows that you want to compare
    x_quant_0 = df.iloc[start_row : end_row+1 : 3]
    x_quant_0 = x_quant_0.append(df.loc[0])
    x_quant_0 = x_quant_0.sort_values(col)

    x_quant_1 = df.iloc[start_row+1 : end_row+2 : 3]
    x_quant_1 = x_quant_1.append(df.loc[1])
    x_quant_1 = x_quant_1.sort_values(col)

    x_quant_2 = df.iloc[start_row+2 : end_row+3 : 3]
    x_quant_2 = x_quant_2.append(df.loc[2])
    x_quant_2 = x_quant_2.sort_values(col)

    # plot the line chart
    plt.plot(np.arange(len(x_quant_0[col])), x_quant_0['acc'], c='b', label = 'ht not quantized', linewidth = 1, linestyle='-', marker='.')
    plt.plot(np.arange(len(x_quant_1[col])), x_quant_1['acc'], c='r', label = 'ht quantized 1', linewidth = 1, linestyle='-', marker='.')
    plt.plot(np.arange(len(x_quant_2[col])), x_quant_2['acc'], c='g', label = 'ht 1uantized 2', linewidth = 1, linestyle='-', marker='.')

    plt.xlabel(x_label)
    plt.xticks(np.arange(len(x_quant_0[col])), x_quant_0[col])
    plt.ylabel('Accuracy [%]')
    plt.ylim(0.65, 1.00)
    plt.yticks([5*i/100 + 0.70 for i in range(6)])
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=10)

    plt.title(plt_title)


    # SAVE GRAPH IN PDF
    filename = os.path.basename(file_path) # Extract filename
    root, ext = os.path.splitext(filename)  # split filename and extension
    parts = root.split('-')    # Split filename into components

    parts[0] = "graphs"    # Modify the desired part
    dir_name = '-'.join(parts[:])   # Join the parts back into a string with the directory name

    parts[0] = "graph"    # Modify the desired part
    parts.append(col)
    graph_name = '-'.join(parts[:]) + '.pdf'  # create new filename with new extension

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(os.path.join(dir_name, graph_name), bbox_inches='tight',dpi=300)

    return plt


df = parse_csv()

plt = generate_plot_num_axis(df=df, 
                    start_row=36, 
                    end_row=53, 
                    col='size_ht',
                    x_label='size ht',
                    x_ticks=list(range(16,129,16)), 
                    plt_title='Comparison of accuracy for different size_ht values')
plt.show()
plt.clf()

plt = generate_plot_num_axis(df=df, 
                            start_row=6, 
                            end_row=11, 
                            col='num_glimpses', 
                            x_label='num glimpses', 
                            x_ticks=[5,10,15,20], 
                            plt_title='Comparison of accuracy for different num glimpses values')
plt.show()
plt.clf()

plt = generate_plot_string_axis(df=df, 
                                start_row=12, 
                                end_row=20, 
                                col='patch_size', 
                                x_label='patch size', 
                                plt_title='Comparison of accuracy for different patch size values')
plt.show()
plt.clf()

plt = generate_plot_num_axis(df=df, 
                            start_row=21, 
                            end_row=29, 
                            col='glimpse_scale', 
                            x_label='glimpse scale', 
                            x_ticks=[2,3,4,5], 
                            plt_title='Comparison of accuracy for different glimpse scale values')
plt.show()
plt.clf()

plt = generate_plot_num_axis(df=df, 
                    start_row=30, 
                    end_row=35, 
                    col='num_patches', 
                    x_label='num patches', 
                    x_ticks=[2,3,4], 
                    plt_title='Comparison of accuracy for different patches number values')
plt.show()
plt.clf()

plt = generate_plot_num_axis(df=df, 
                    start_row=54, 
                    end_row=65, 
                    col='phi_train', 
                    x_label='patch quantization', 
                    x_ticks=[1,2,3,4,5,6,7,8], 
                    plt_title='Comparison of accuracy for different patch quantization values')
plt.show()
plt.clf()

def generate_plot_hidden_quant(df, col, x_label, x_ticks, plt_title):
    # filter the rows that you want to compare
    x_quant_0 = df.iloc[0:6]

    # plot the line chart
    plt.plot(x_quant_0[col], x_quant_0['acc'], c='b', linewidth = 1, linestyle='-', marker='.')

    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    plt.ylabel('Accuracy [%]')
    plt.ylim(0.65, 1.00)
    plt.yticks([5*i/100 + 0.70 for i in range(6)])
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=10)

    plt.title(plt_title)


    # SAVE GRAPH IN PDF
    filename = os.path.basename(file_path) # Extract filename
    root, ext = os.path.splitext(filename)  # split filename and extension
    parts = root.split('-')    # Split filename into components

    parts[0] = "graphs"    # Modify the desired part
    dir_name = '-'.join(parts[:])   # Join the parts back into a string with the directory name

    parts[0] = "graph"    # Modify the desired part
    parts.append(col)
    graph_name = '-'.join(parts[:]) + '.pdf'  # create new filename with new extension

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(os.path.join(dir_name, graph_name), bbox_inches='tight',dpi=300)

    return plt

plt = generate_plot_hidden_quant(df=df,
                                 col='ht_test',
                                 x_label='ht quantization',
                                 x_ticks=[0,1,2,3,4,5,6,7,8],
                                 plt_title='Comparison of accuracy for different hidden vector quant values'
                                 )
                                 
plt.show()
plt.clf()