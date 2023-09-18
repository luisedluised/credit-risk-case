import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from skopt import BayesSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


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

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,  numerical_features, categorical_features, dropped_features = []):
        self.dropped_features = dropped_features
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
    
    def fit(self, X, y=None):
        self.categorical_features_order = sort_categorical_values_by_correlation(
            X.copy(), y, self.categorical_features)

        self.positive_and_negative_indicators_frme = find_positive_and_negative_indicators(
            X.copy(), y, self.numerical_features, self.categorical_features)
        return self

    def transform(self, X):
        X = encode_categorical_values(X.copy(), self.categorical_features_order)
        X = apply_indicators_conditions(X.copy(), self.positive_and_negative_indicators_frme)
        X = X.drop(self.dropped_features, axis = 1)
        self.features = X.columns
        return X


### receives a model and a parameter range and uses bayes search to find the best parameters and scores
def model_testing_pipeline(model, model_params, cv, n_iter, X_train, y_train, 
    numerical_features, categorical_features, dropped_features = [], print_scores = True):

    pipeline = Pipeline([
        ('preprocess', PreprocessingTransformer(numerical_features, categorical_features, dropped_features)), 
        ('model', model)])

    bayes_search = BayesSearchCV(
        pipeline,
        model_params,
        cv=cv,
        verbose=-1,
        n_jobs=-1,
        return_train_score=True,
        n_iter = n_iter,
        scoring='roc_auc')

    bayes_search.fit(X_train, y_train)

    best_estimator_test_score = bayes_search.best_score_
    best_estimator_train_score = bayes_search.cv_results_['mean_train_score'][bayes_search.best_index_]

    output = {
        'best_estimator_test_score': best_estimator_test_score,
        'best_estimator_train_score': best_estimator_train_score,
        'best_estimator_test_score_std': bayes_search.cv_results_['std_test_score'][bayes_search.best_index_],
        'best_estimator': bayes_search.best_estimator_,
        'best_params': bayes_search.best_params_}
        
    warn(model_params, bayes_search.best_params_)

    if print_scores == True:
        print('best estimator test score:', best_estimator_test_score)
        print('best estimator train score:', best_estimator_train_score)

    return output, bayes_search


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


def backward_elimination_round(n_evaluations, model, model_params, all_features, previously_removed, X_train, y_train, numerical_features, categorical_features):
    rows = []

    all_tested_feats = [x for x in all_features if x not in previously_removed]

    for i in range(n_evaluations):
        for tested_feature in (all_tested_feats + [[]])[::-1]:
            feature_formatted = tested_feature if type(tested_feature) == list else [tested_feature]
            dropped = list(set(feature_formatted).union(set(previously_removed)))
            
            pipeline = Pipeline([
                ('preprocess', PreprocessingTransformer(numerical_features, categorical_features, dropped)), 
                ('model', model)])
            xtrain, xtest, ytrain, ytest = train_test_split(
                X_train, y_train, test_size=0.2, stratify = y_train)
            pipeline.fit(xtrain, ytrain)
            ypred = pipeline.predict_proba(xtest)[:,1]
            auc = roc_auc_score(ytest, ypred)

            res = (pd.DataFrame({
                    'removed_feature': str(tested_feature),
                    'test_score': auc}, index=[0])
                    )

            rows.append(res)

    results = pd.concat(rows).reset_index(drop=True)

    results.removed_feature = results.removed_feature.astype(str)
    results['avg_test_score'] = results.groupby('removed_feature').test_score.transform('mean')
    results['avg_std_test_score'] = results.groupby('removed_feature').test_score.transform('std') / np.sqrt(n_evaluations)
    results = results.sort_values('avg_test_score', ascending=False).drop_duplicates('removed_feature').reset_index(drop=True)
    results = results.drop('test_score', axis=1)

    worse = results.iloc[0]
    worse_feature = worse.removed_feature

    if worse_feature == '[]':
        print('Finished.')
        return '[]', results

    else:
        no_removed = results[results.removed_feature == '[]'].iloc[0]
        no_removed_score = no_removed.avg_test_score
        worse_score = worse.avg_test_score

        std1 = worse.avg_std_test_score 
        std2 = no_removed.avg_std_test_score 
        difference_std = (std1**2 + std2**2)**0.5
        
        if np.abs(no_removed_score - worse_score) < 2 * difference_std:
            print('Finished.')
            return '[]', results
        else:
            print('Removing', worse_feature)
            return worse_feature, results


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
        'gold' if x == res.expected_return.max() else 'silver')
                
    return res


def model_returns_estimation(bad_client_incidence, bad_client_cost, specificity = 0.3, sensitivity = 0.3):
    if bad_client_cost < 0:
        bad_client_cost = -bad_client_cost*-1

    expected_bad_clients = 100*bad_client_incidence
    expected_good_clients = 100*(1-bad_client_incidence)

    lost_good = expected_good_clients * (1 - specificity)
    avoided_bad = expected_bad_clients * sensitivity

    expected_return = avoided_bad * bad_client_cost - lost_good

    percentual_expected_return = expected_return/expected_good_clients

    return(percentual_expected_return)