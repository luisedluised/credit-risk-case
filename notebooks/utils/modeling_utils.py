import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def find_positive_and_negative_indicators(df, y, numerical_features, categorical_features):
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


def sort_categorical_values_by_correlation(df, y, categorical_features):
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
    categorical_features = categorical_encoder_frame.feature.unique()
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


def get_metrics_by_threshold_table(y_test, y_pred_proba, bad_customer_cost_ratio):
    n_good_clients = (y_test == 0).sum()
    n_bad_clients = (y_test == 1).sum()

    res = pd.DataFrame()
    for threshold in np.linspace(0.0, 1.0, 11).tolist():
        pred = (y_pred_proba >= threshold).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        tn = ((pred == 0) & (y_test == 0)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()

        precision = tp / (tp + fp)
        specificity = tn / n_good_clients
        sensitivity = tp / n_bad_clients

        row = pd.DataFrame({ 
            'thres': threshold, 'precision': 100*precision, 'recall/sensitivity': 100*sensitivity,
            'specificity': 100*specificity, 'lost_good_clients': fp, 'avoided_bad_clients': tp
            }, index=[0])

        res = pd.concat([res, row]).reset_index(drop=True)
        if sensitivity == 0: break

    res = res.round(2)

    res['loss'] = res.lost_good_clients 
    res['gain'] = res.avoided_bad_clients*bad_customer_cost_ratio
    res['expected_return'] = res.gain - res.loss

    res['bottom_line'] = (n_good_clients - res.lost_good_clients 
                                    - bad_customer_cost_ratio*(n_bad_clients - res.avoided_bad_clients))

    res['color'] = res.expected_return.apply(lambda x: 
        'gold' if x == res.expected_return.max() else 'gray')
                
    return res