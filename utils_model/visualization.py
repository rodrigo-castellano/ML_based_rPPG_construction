import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from statannotations.Annotator import Annotator


from utils_model.data_helper import *




''' VISUALIZATION FUNCTIONS '''

def get_pairs_pvalues(p_values, pairs):
    """
    Returns the pairs of indices where the p-value is greater than 0.05
    
    Parameters:
        p_values (list): A list of p-values
        pairs (list): A list of pairs
        
    Returns:
        pairs_pvalues (list): A list of pairs of indices where the p-value is greater than 0.05
        p_vals (list): A list of p-values
    """
    pairs_pvalues = []
    p_vals = []
    for i in range(len(p_values)):
        pairs_pvalues.append(pairs[i])
        p_vals.append(p_values[i])
    return pairs_pvalues, p_vals

def get_pairs(data_pvalues, items, methods, dataset):
    """
    Returns a list of pairs for a given dataset
    
    Parameters:
        data_pvalues (DataFrame): A Pandas DataFrame containing the p-values
        items (list): A list of items
        methods (list): A list of methods
        dataset (str): A string representing the name of the dataset
        
    Returns:
        pairs (list): A list of pairs for a given dataset
    """
    k = 0
    pairs = []
    for n in range(len(methods) - 1):
        dat_name = dataset
        meth_name = data_pvalues.index[k][1].split()[0]
        meth_name = meth_name.split('_')[0]
        pair1 = (dat_name + '_' + meth_name)
        pair2 = (dat_name + '_' + 'Model')
        pairs.append((pair1, pair2))
        k += 1
    return pairs


def plot_results_RGB(data_pvalues,data_table,items,metrics,figures_path,my_font,mode='activities',type='bar',save=False):
    """Plots the results of an RGB analysis by item and metric.

    Args:
        data_pvalues (pandas.DataFrame): The p-values for each pair of color channels and metric.
        data_table (pandas.DataFrame): The data table with the metric values for each item and channel.
        items (list): The items to plot.
        metrics (list): The metrics to plot.
        figures_path (str): The path where to save the generated figures.
        my_font (str): The font to be used for the plot.
        mode (str, optional): The mode to use for the plot. Defaults to 'act'.

    Returns:
        None
    """
    prop = fm.FontProperties(fname=".\\other\\fonts\\times-ro.ttf") # Font to be used
    colors = ['Red','Green','Blue']
    palette = ['tomato','lightgreen','lightblue']
    # -- PLOT THE RESULTS BY items
    # fig = plt.figure() 
    plt.rcParams["figure.figsize"] = (20,6)
    plt.rcParams.update({'font.size': 15})

    # Set the metric to plot
    for metric in metrics:
        # Create the tuples with the pairs of channels for the p-values
        pairs = []
        for i in range(len(list(items))):
            for j in range(3):
                color0 =data_pvalues.index[3*i+j][1].split()[0]
                color1 =data_pvalues.index[3*i+j][1].split()[2]
                a = data_pvalues.index[3*i+j][0]+'_'+color0
                b = data_pvalues.index[3*i+j][0]+'_'+color1
                pairs.append((a,b))
        # create an array with the p-values
        p_values = np.abs(data_pvalues[[metric]].T.values[0])

        # This is just to give the annotator class (below) the number of pairs and their name
        nn = {}
        for i,name in enumerate(data_table.index):
            nn[name[0]+'_'+name[1]] = [1]
        nn = pd.DataFrame(nn)
        # -- PLOT THE DATA AND THE P-VALUES
        if type=='boxplot':
            ax = sns.boxplot(data=data_table.loc[items][[metric]].T.values[0], palette = palette, showfliers=False,width=.3,linewidth=1.1,boxprops=dict(alpha=1))
        elif type=='bar':
            ax = sns.barplot(data=data_table.loc[items][[metric]].T.values[0], palette = palette,width=.3,linewidth=1.1)

        annotator = Annotator(ax, pairs, data=pd.DataFrame(nn))
        annotator.configure(text_format="simple", loc="inside",verbose=False,fontsize=22)
        annotator.set_pvalues_and_annotate(p_values)

        ax.set_xticks([*range(1,len(list(items))*3+1,3)])    
        ax.set_xticklabels(items, fontproperties=my_font,fontsize=32)
        # plt.xlabel('Dataset', fontproperties=my_font,fontsize=32, labelpad=2)
        if metric=='r':
            plt.ylabel('$r$', fontproperties=my_font,fontsize=32)
            plt.title('$r$ coefficient' +' across '+mode, x=0.5, y=1.05, fontproperties=my_font,fontsize=32)
        else:
            plt.ylabel(metric, fontproperties=my_font,fontsize=32)
            plt.title(metric+' across '+mode, x=0.5, y=1.05, fontproperties=my_font,fontsize=32)
        fig = plt.gcf() # This is to be able to save and show the figure
        if save:
            fig.savefig(figures_path+mode+'_'+metric+'.png',dpi=300)
        plt.show()

        

def set_custom_palette(series, max_color='turquoise', other_color='lightblue', mode='min'):
    """
    Generate a custom color palette based on a pandas series.
    
    Args:
    - series (pandas series): the series of values to base the color palette on
    - max_color (str): the color to use for the maximum value in the series (default: 'turquoise')
    - other_color (str): the color to use for all other values in the series (default: 'lightblue')
    - mode (str): the mode to use for selecting the best value in the series ('min' or 'max') (default: 'min')
    
    Returns:
    - pal (list): a list of colors corresponding to the input series, with the color for the best value set to max_color
    
    """
    
    # Select the best value based on the input mode
    if mode == 'max':
        best_val = series.max()
    elif mode =='min': 
        best_val = series.min()
    
    # Generate the color palette
    pal = []
    for item in series:
        if item == best_val:
            pal.append(max_color)
        else:
            if other_color == 'purple':
                # Use an xkcd color if other_color is set to purple
                pal.append(sns.xkcd_rgb['orchid'])
            else:
                pal.append(other_color)
    
    return pal




def get_pvalues_ML(data, items, metrics, methods, table):
    """
    Calculate p-values for a set of metrics using Friedman test followed by Nemenyi post-hoc test.

    Args:
    - data (pandas DataFrame): input data
    - items (list): items to perform the tests on (e.g., datasets or activities)
    - metrics (list): metrics to perform the tests on
    - methods (list): methods used to compute the metrics, where the last method is considered as the model to compare with
    - table (pandas DataFrame): data table with the metrics and methods used for each item

    Returns:
    - items_pvalues (pandas DataFrame): p-values for all the metrics and comparisons between the model and the other methods for each item
    """

    # Perform Friedman test followed by the post-hoc Nemenyi test for the three channels
    tests_metrics = []
    for metric in metrics:
        tests_items = []
        for item in items:
            data_item = []
            for method in methods:  
                data_item.append(table[[metric]].loc[item].loc[method].values[0])
            _,  p_value = friedmanchisquare(*data_item)

            # If p-value > 0.05, the difference between the methods is not significant
            if p_value > 0.05:
                print('P-value not significant for', metric, 'in', item)
                test = [p_value for i in range(len(methods)-1)]
            else: 
                # If the p-value is significant, perform Nemenyi test
                nemenyi = sp.posthoc_nemenyi_friedman(np.array(data_item).T)  
                # select the column from results corresponding to the method Model, which is the last one
                test = nemenyi.values[-1,:-1]
            tests_items.extend(test)
        tests_metrics.append(tests_items)

    # Convert the results to a dataframe
    ind = [name +' vs. Our Model' for name in methods[:-1]] 
    mux = pd.MultiIndex.from_product([list(items), ind])
    items_pvalues = pd.DataFrame(np.array(tests_metrics).T, columns=metrics).set_axis(mux).round(3)
    return items_pvalues


def get_pvalues_RGB(data: pd.DataFrame, items: list, metrics: list, methods: list, table: pd.DataFrame) -> pd.DataFrame:
    """Perform Friedman test followed by the post-hoc Nemenyi test for RGB channels.

    Args:
        data (pd.DataFrame): The data used for testing.
        items (list): A list of items (e.g., datasets or activities) to be tested.
        metrics (list): A list of metrics to be tested.
        methods (list): A list of methods to be tested.
        table (pd.DataFrame): A table of the results of the methods for each item and metric.

    Returns:
        pd.DataFrame: A dataframe of p-values for RGB channel comparisons.
    """
    # Perform Friedman test followed by the post-hoc Nemenyi test for the three channels
    tests_metrics = []
    for metric in metrics:
        tests_items = []
        for item in items:
            data_item = []
            for method in methods:
                data_item.append(table[[metric]].loc[item].loc[method].values[0])
            # Perform Friedman test for the data
            _,  p_value = friedmanchisquare(*data_item)
            if p_value > 0.05:
                # If p-value is not significant, assign all p-values to be the p-value of the Friedman test
                print('P-value not significant for', metric, 'in', item)
                test = [p_value for i in range(len(methods))]
            else: 
                # If the p-value is significant, perform Nemenyi test
                nemenyi = sp.posthoc_nemenyi_friedman(np.array(data_item).T)  
                # Get p-values for comparisons between red vs green, red vs blue, and green vs blue
                r_vs_g = nemenyi.iloc[1, 0]
                r_vs_b = nemenyi.iloc[2, 0]
                g_vs_b = nemenyi.iloc[2, 1]
                test = [r_vs_g, r_vs_b, g_vs_b]
            tests_items.extend(test)
        tests_metrics.append(tests_items)

    # Convert it to dataframe and save it
    ind = ['Red vs Green', 'Red vs Blue', 'Green vs Blue']
    mux = pd.MultiIndex.from_product([list(items), ind])
    items_pvalues = pd.DataFrame(np.array(tests_metrics).T, columns=metrics).set_axis(mux).round(4)
    return items_pvalues

def get_table_results(data, items, metrics, methods, methods_names):
    """
    Creates a table of results for multiple datasets/activities and methods.

    Args:
        data (dict): A dictionary containing the results of experiments. 
                     The keys of the dictionary are the names of the datasets/activities 
                     and the values are dictionaries with the metrics as keys and 
                     lists of scores as values.
        items (list): A list of dataset/activity names to include in the table.
        metrics (list): A list of metric names to include in the table.
        methods (list): A list of method names in the order they should appear in the table.
        methods_names (list): A list of display names for the methods, in the same order as the methods list.

    Returns:
        tuple: A tuple of two dataframes: 
               1. The first dataframe is a multi-index table containing the raw results of the experiments, 
                  with one row per method and one column per metric, for each dataset/activity.
               2. The second dataframe is a multi-index table containing the average results for each method and metric, 
                  rounded to two decimal places.

    Example:
        data = {'dataset1': {'accuracy': [0.8, 0.9, 0.7], 'precision': [0.7, 0.8, 0.6]}, 
                'dataset2': {'accuracy': [0.6, 0.5, 0.4], 'precision': [0.6, 0.4, 0.5]}}
        items = ['dataset1', 'dataset2']
        metrics = ['accuracy', 'precision']
        methods = ['method1', 'method2', 'method3']
        methods_names = ['Method 1', 'Method 2', 'Method 3']
        table, table_avg = get_table_results(data, items, metrics, methods, methods_names)
    """
    dfs = []
    for item in items:
        # create a dataframe for each item and metric
        df = pd.DataFrame(data[item], columns=metrics, index=methods)
        # put the methods in the order given
        df_methods = df.loc[methods]
        dfs.append(df_methods)
    # concatenate the dataframes for all items into a single table
    mux = pd.MultiIndex.from_product([list(items), methods_names])
    table = pd.concat(dfs, axis=0).set_axis(mux, axis=0)
    # compute the average results for each method and metric
    table_avg = table.applymap(avg_cell).round(2)
    return table, table_avg
