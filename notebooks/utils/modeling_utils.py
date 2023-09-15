import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def find_positive_and_negative_indicators(df, y):
    df['target'] = y
    top_quantiles_space = np.linspace(.1, 0.9, 9)

    numerical_frame = pd.DataFrame()
    for feature in numerical_features:
        for q in top_quantiles_space:
            value = df[feature].quantile(q)

            evaluation_series = (df[feature] >= value)
            r, p = pearsonr(df.target, evaluation_series)
            numerical_frame = pd.concat([numerical_frame, pd.DataFrame({'feature': feature, 'operation':'bigger than', 'quantile': q, 'pearsonr': r, 'value':value}, index=[0])])
            
            evaluation_series = (df[feature] <= value)
            r, p = pearsonr(df.target, evaluation_series)
            numerical_frame = pd.concat([numerical_frame, pd.DataFrame({'feature': feature, 'operation':'smaller than', 'quantile': q, 'pearsonr': r, 'value':value}, index=[0])])

    categorical_frame = pd.DataFrame()
    for feature in categorical_features:
        for category in df[feature].unique():
            evaluation_series = (df[feature] == category)
            r, p = pearsonr(df.target, evaluation_series)
            categorical_frame = pd.concat([categorical_frame, pd.DataFrame({'feature': feature, 'value': category, 'pearsonr': r}, index=[0])])

    ## positive_indicators selection
    numerical_negative_indicators = numerical_frame[numerical_frame.pearsonr > 0.1]
    numerical_negative_indicators = numerical_negative_indicators.sort_values('quantile').drop_duplicates(['feature', 'operation'], keep='first')
    categorical_negative_indicators = categorical_frame[categorical_frame.pearsonr > 0.1]
    negative_indicators = pd.concat([numerical_negative_indicators, categorical_negative_indicators])

    numerical_positive_indicators = numerical_frame[numerical_frame.pearsonr < -0.1]
    numerical_positive_indicators = numerical_positive_indicators.sort_values('quantile').drop_duplicates(['feature', 'operation'], keep='first')
    categorical_positive_indicators = categorical_frame[categorical_frame.pearsonr < -0.1]
    positive_indicators = pd.concat([numerical_positive_indicators, categorical_positive_indicators])

    positive_indicators['type'] = 'positive_indicator'
    negative_indicators['type'] = 'negative_indicator'

    indicators_conditions = pd.concat([positive_indicators, negative_indicators])
    indicators_conditions = indicators_conditions[['feature','operation','value', 'type']]
    indicators_conditions.operation = indicators_conditions.operation.fillna('equal to')
        
    return indicators_conditions

def apply_indicators_conditions(df, indicators_conditions):
    df['number_positive_indicators'] = 0
    df['number_negative_indicators'] = 0
    for condition in indicators_conditions.iloc:
        indicator_type = condition.type ### good or bad
        if condition.operation == 'bigger than':
            df[f'number_{indicator_type}s'] += (df[condition.feature] >= condition.value).astype(int)
        elif condition.operation == 'smaller than':
            df[f'number_{indicator_type}s'] += (df[condition.feature] <= condition.value).astype(int)
        else:
            df[f'number_{indicator_type}s'] += (df[condition.feature] == condition.value).astype(int)
            
    df['positive_negative_balance'] = df.number_positive_indicators - df.number_negative_indicators
    return df


def sort_categorical_values_by_correlation(df, y):
    df['target'] = y
    categorical_encoder_frame = pd.DataFrame()
    for feature in categorical_features:
        for value in df[feature].unique():
                evaluation_series = (df[feature] == value)
                r, p = pearsonr(df.target, evaluation_series)
                categorical_encoder_frame = pd.concat(
                    [categorical_encoder_frame, pd.DataFrame(
                        {'feature': feature, 'value': value, 'pearsonr': r}, index=[0])])
                        
    categorical_encoder_frame = categorical_encoder_frame.sort_values(['feature','pearsonr']
        ).drop('pearsonr', axis=1).reset_index(drop=True)
    categorical_encoder_frame['encoded'] = 1
    categorical_encoder_frame.encoded = categorical_encoder_frame.groupby('feature').encoded.cumsum()
    return categorical_encoder_frame

def encode_categorical_values(df, categorical_encoder_frame):
    for feature in categorical_features:
        encoding_dic = categorical_encoder_frame[categorical_encoder_frame.feature == feature
            ][['value', 'encoded']].set_index('value').to_dict()['encoded']
        df[feature] = df[feature].map(encoding_dic)
    return df


def warn(model_params, best_params):
    for param in model_params:
        prior = model_params[param].prior
        if 'prior' not in ['uniform', 'log-uniform']:
            continue

        low = model_params[param].low
        high = model_params[param].high

        if prior == 'log-uniform':
            space = np.logspace(np.log10(low), np.log10(high), 10)
        elif prior == 'uniform':
            space = np.linspace(low, high, 10)
        value_found = best_params[param]

        if (value_found > space[-2]) or (value_found < space[1]):
            print('warning:', param, 'is at the edge of the search space (value:', value_found, ')')



def selection_round(n_evaluations, model, model_params, all_features, previously_removed):
    rows = []

    all_tested_feats = [x for x in all_features if x not in previously_removed]

    for i in range(n_evaluations):
        for tested_feature in (all_tested_feats + [[]])[::-1]:
            feature_formatted = tested_feature if type(tested_feature) == list else [tested_feature]
            dropped = list(set(feature_formatted).union(set(previously_removed)))
            res, _ = test_pipeline(model, model_params, 5, 5, dropped_features=dropped, print_scores=False)
            res = (pd.DataFrame({
                    'removed_feature': str(tested_feature),
                    'test_score': res['best_estimator_test_score']}, index=[0])
                    )
            rows.append(res)

    results = pd.concat(rows).reset_index(drop=True)

    results.removed_feature = results.removed_feature.astype(str)
    results ['avg_test_score'] = results.groupby('removed_feature').test_score.transform('mean')
    results = results.sort_values('avg_test_score', ascending=False).drop_duplicates('removed_feature').reset_index(drop=True)
    worse = results.iloc[0]
    worse_feature = worse.removed_feature


    if worse_feature == '[]':
        print('acabou')
        return '[]', results

    print('removing', worse_feature)
    return worse_feature, results