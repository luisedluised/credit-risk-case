import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

from utils.modeling_utils import *

def format_number(number):
    if number > 1000:
        return f'{number/1000:.0f}k'
    else:
        return f'{number:.0f}'
    
def kde(feature, df):
    fig, ax = plt.subplots(figsize=(7, 2))
    sns.kdeplot(df[df.risk == 'bad'][feature], label='bad', ax=ax, color = 'maroon', shade = True, alpha = 0.99, linewidth = 0.1)
    sns.kdeplot(df[df.risk == 'good'][feature], label='good risk', ax=ax, color = 'silver', shade = True, alpha = 0.5, linewidth = 0.9)
    plt.ylabel('density')
    plt.yticks([])
    plt.legend()
    ## hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def stripplot(categorical_feature, numerical_feature, df, figsize=(12, 3)):
    fig, ax = plt.subplots(figsize = figsize)

    sns.violinplot(x=categorical_feature, y=numerical_feature,
                    data=df, ax=ax, color=0.85*np.ones(3), linewidth=0, width=1)

    sns.stripplot(x=categorical_feature, y=numerical_feature,
                    data=df, ax=ax, color='black', alpha = 0.6, size=3, jitter=0.2)

    yticks = np.linspace(df[numerical_feature].min(), df[numerical_feature].max(), 6)
    ax.set_yticks(yticks);
    ax.set_yticklabels([format_number(x) for x in yticks]);

    xticks = df[categorical_feature].unique().tolist()
    ax.set_xticklabels([str(x)[:20] for x in xticks], fontsize = 9, color = 'dimgray');

    ## alggn to the left
    ax.set_xlabel(categorical_feature, ha='left', x=0);
    ax.set_ylabel(numerical_feature);

    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis='x', bottom=False)

    
def plot_stacked_risk(input_frame, feature, is_numerical = False, ax = None, hide_labels = False, figsize = (10, 3)):
    df = input_frame.copy()
    df[feature] = df[feature].fillna('undefined')

    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    average_risk = df.risk.value_counts(normalize = True).bad

    if is_numerical:
        df['temp'] = df[feature]
        df[feature] = pd.qcut(df[feature], 4, labels=False, duplicates='drop')
        df = df.sort_values(feature)
        xticklabels =  df.groupby(feature)['temp'].apply(lambda x: 
                                                format_number(x.min()) +  ' to '+ format_number(x.max())).tolist()
        
    values = df[feature].unique()

    weight_counts = {'bad': [], 'good': []}
    errors = []

    for i, value in enumerate(values):
        value_frame = df[df[feature] == value].risk.value_counts(normalize = True
                        ).reset_index().rename({'risk':f'risk{i}'}, axis = 1)

        if i == 0: frame = value_frame
        else: frame = frame.merge(value_frame, 'outer','index')

        try:
            p = value_frame[f'risk{i}'].values[0]
            q = value_frame[f'risk{i}'].values[1]
            N = df[df[feature] == values[0]].shape[0]
            se = np.sqrt(p*q/N)
            errors.append(se)
        except:
            print(feature, value)
            
    frame = frame[['index'] + [x for x in frame.columns if x != 'index']].fillna(0.5)
    frame['index'] = frame['index'].astype(str)
    for row in frame.iloc:
        v = row.tolist()
        weight_counts[v[0]] = 100*np.array(v[1:])

    colors = ['maroon', 'silver']
    bottom = np.zeros(len(values))
    for i, (label, weight_count) in enumerate(weight_counts.items()):
        label = 'good risk' if label == 'good' else label
        yerr = 100*np.array(errors) if label == 'bad' else None

        p = ax.bar(values, weight_count, 0.65, label=label, bottom=bottom, color = colors[i]
                   , yerr=yerr, error_kw=dict(ecolor='white', lw=2.5, capsize=0, capthick=2.5,
                                              alpha=0.99))
        bottom += weight_count

    #plot horizontal line
    ax.axhline(y=100*average_risk, color='black', linestyle='-', label='dataset average', linewidth=1.5, alpha = 0.6)

    if not hide_labels:
        ax.legend(loc='upper center', bbox_to_anchor=(0.8, 1.3),
                    ncol=3, fancybox=True, shadow=False
                    ,frameon=True, fontsize=9)
        ax.set_ylabel('percentual target')
        
    if is_numerical: s = ' quarters'
    else: s = ''

    ax.set_title(feature + s, fontsize=14, fontweight='bold', pad=20, loc='left', color = (0.3,0.1,0.1))

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_color('gray')
    ax.tick_params(axis='y', which='both', length=5, labelright=True, labelleft=False, right=False, left=False)
    ax.tick_params(axis='x', which='both', length=0, labelrotation=0, labelsize=9)
    ax.yaxis.set_label_position("right")


    if is_numerical:
        ax.set_xticks(values)
        ax.set_xticklabels(xticklabels, rotation=0, fontsize=9)

def plot_roc_and_pr_curves(fpr, tpr, recall, precision, class_imbalance, figsize = (13, 3)):
    _, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(recall, precision, label = 'model')
    ax[0].plot([0, 1], [class_imbalance, class_imbalance], 'k--', label = 'no skill')
    ax[0].fill_between(recall, precision, 0, alpha=0.05)
    ax[0].fill_between([0, 1], [class_imbalance, class_imbalance], color = 'white')
    ax[0].set_xlabel('recall')
    ax[0].set_ylabel('precision')
    ax[0].set_title('precision recall curve')
    ax[0].yaxis.tick_right()
    ax[0].set_ylim([-0.05, 1.05])
    ax[0].legend()

    ax[1].plot(fpr, tpr, label = 'model')
    ax[1].plot([0, 1], [0, 1], 'k--', label = 'no skill')
    ax[1].fill_between(fpr, tpr, 0, alpha=0.05)
    ax[1].fill_between([0, 1], [0, 1], color = 'white')
    ax[1].legend()


    ax[1].set_xlabel('false positive rate')
    ax[1].set_ylabel('true positive rate')
    ax[1].set_title('roc curve')
    ax[1].yaxis.set_label_position("right")



def plot_default_and_loss(specificity, sensitivity, 
         bad_client_incidence_range = (0.01, 0.99), bad_client_cost_range = (-5, 10),
         highlighted_points = False, resolution = 200, clip = 100):

    x, y, c = [], [], []
    r, g, b = [], [], []

    X = np.linspace(bad_client_incidence_range[0], bad_client_incidence_range[1], resolution)
    Y = np.logspace(np.log10(bad_client_cost_range[0]), np.log10(bad_client_cost_range[1]), resolution)

    for i in X:
        for j in Y:
            expected_return = model_returns_estimation(i, j, specificity = specificity, sensitivity = sensitivity)
            blue = np.clip(expected_return, 0, clip)
            red = - np.clip(expected_return, -clip, 0)

            x.append(i), y.append(j), r.append(red), b.append(blue)

    r = np.array(r)/np.max(r)
    b = np.array(b)/np.max(b)

    b = (b**0.6)*0.7
    r = (r**0.6)*0.7
    g = b*0.4 + r*0.15

    c = np.array([r, g, b]).T

    plt.figure(figsize=(10.,5))
    plt.scatter(x, y, c = c, alpha = 0.25, s = 65, marker = 'o')
    ax = plt.gca()

    ax.set_xlim(bad_client_incidence_range)
    ax.set_ylim(bad_client_cost_range)

    plt.yscale("log")
    yticks = np.logspace(np.log10(bad_client_cost_range[0]), np.log10(bad_client_cost_range[1]), 6)
    ax.set_yticks(yticks);
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_yticklabels([round(x, 2) for x in yticks]);
    

    plt.title('Expected model return as a function of bad_client_incidence & bad_client_cost', fontsize = 10)

    plt.xlabel('bad_client_incidence')
    plt.ylabel('bad_client_cost')
                
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.yaxis.set_label_position("right")
    
    if highlighted_points is not False:
        plt.scatter(highlighted_points[0], highlighted_points[1], c = 'gold', s = 6, marker = 'o', alpha = 0.8)

    ax.annotate('higher $ loss', (0.03, 0.05), xycoords = 'axes fraction',
                fontsize = 8, color = 'white', alpha = 1)
    ax.annotate('higher $ returns', (0.85, 1 - 0.05), xycoords = 'axes fraction',
                    fontsize = 8, color = 'white', alpha = 1)


def plot_specificity_sensitivity(bad_client_incidence, bad_client_cost, 
         specificity_range = (0.01, 0.99), sensitivity_range = (0.01, 0.99),
         highlighted_points = False, resolution = 200, clip = 100):

    x, y, c = [], [], []
    r, g, b = [], [], []

    for i in np.linspace(specificity_range[0], specificity_range[1], resolution):
        for j in np.linspace(sensitivity_range[0], sensitivity_range[1], resolution):
            x.append(i)
            y.append(j)

            expected_return = model_returns_estimation(bad_client_incidence, bad_client_cost, specificity = i, sensitivity = j)

            blue = np.clip(expected_return, 0, clip)
            red = np.clip(expected_return, -clip, 0) * -1
            green = 0

            r.append(red)
            g.append(green)
            b.append(blue)

    r = np.array(r)/np.max(r)
    b = np.array(b)/np.max(b)

    r = (r**0.6)*0.7
    b = (b**0.6)*0.7
    g = b*0.4 + r*0.15

    c = np.array([r, g, b]).T

    plt.figure(figsize=(10.,5))
    plt.title('Expected model return as a function of sensitivity & sensibility. '
        'Dots represent different classification threshold choices', fontsize = 9)
    
    plt.scatter(x, y, c = c, alpha = 0.3, s = 65,
                marker = 'o')

    plt.xlabel('specificity')
    plt.ylabel('sensitivity')

    ax = plt.gca()
    ax.set_xlim(specificity_range)
    ax.set_ylim(sensitivity_range)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.yaxis.set_label_position("right")

    if highlighted_points is not False:
        plt.scatter(highlighted_points[0], highlighted_points[1],
        c = highlighted_points[3], s = 6, marker = 'o', alpha = 0.8)

    ax.annotate('higher $ loss', (0.03, 0.05), xycoords = 'axes fraction',
                fontsize = 8, color = 'white', alpha = 1)
    ax.annotate('higher $ returns', (0.85, 1 - 0.05), xycoords = 'axes fraction',
                    fontsize = 8, color = 'white', alpha = 1)

    #for i, txt in enumerate(highlighted_points[2]):
    #    ax.annotate(txt, (highlighted_points[0][i], highlighted_points[1][i]),
    #                fontsize = 8, color = 'white', alpha = 1)