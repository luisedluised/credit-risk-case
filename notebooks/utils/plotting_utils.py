import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

## stacked bar plot
#plot_stacked_risk(df, feature)