import typing

from numpy import ndarray
from pandas import DataFrame
from statsmodels.api import add_constant, OLS
from statsmodels.regression.linear_model import RegressionResults

__all__ = [
    'forward_selection',
    'get_formula',
]


def get_model(
        independents: ndarray,
        dependents: ndarray,
) -> RegressionResults:
    """
    Get Ordinary Least Squares Model
    Args:
        independents: independent values
        dependents: dependent values

    Returns:
        Ordinary Least Squares Model
    """
    assert isinstance(independents, ndarray) and len(independents.shape) == 1
    assert isinstance(dependents, ndarray) and len(dependents.shape) == 2
    assert independents.shape[0] == dependents.shape[0]

    dependents = add_constant(dependents, prepend=False)
    return OLS(independents, dependents).fit()


def get_metric(
        model: RegressionResults,
        metric_type: str = 'aic',
) -> float:
    """
    Get Metric
    Args:
        model: RegressionResults
            Ordinary Least Squares Model
        metric_type: str in ['aic', 'bic', 'r2', 'r2_adj'
            'aic' refers Akaike information criterion
            'bic' refers Bayesian information criterion
            'r2' refers R-squared
            'r2_adj' refers Adjusted R-squared

    Returns: float
        metric
    """
    if metric_type == 'aic':
        return model.aic
    elif metric_type == 'bic':
        return model.bic
    elif metric_type == 'r2':
        return model.rsquared
    elif metric_type == 'r2_adj':
        return model.rsquared_adj
    else:
        raise ValueError


def select_dependents(dependents: DataFrame, variates: typing.List[str]) -> ndarray:
    length = dependents.shape[0]
    dependents = dependents.loc[:, variates]
    assert isinstance(dependents, DataFrame)
    dependents = dependents.values
    assert isinstance(dependents, ndarray) and list(dependents.shape) == [length, len(variates)]
    return dependents


def get_formula(
        independent_name: str,
        dependent_names: typing.List[str],
        result: RegressionResults,
) -> str:
    """
    Get Formula
        Format like 'y = w_{0}*x_{0}+w_{1}*x_{1}+...+b'.

    Args:
        independent_name: str
        dependent_names: str
        result: RegressionResults
            Ordinary Least Squares Model

    Returns: str
        Formula
    """
    params = result.params
    return '{}={}+({})'.format(
        independent_name,
        '+'.join('({}*{})'.format(param, name) for param, name in zip(params, dependent_names)),
        params[-1]
    )


def forward_selection(
        independents: DataFrame,
        dependents: DataFrame,
        metric_type: str = 'aic',
        mutex_names: typing.List[tuple] = None,
) -> typing.Tuple[str, typing.List[str], RegressionResults, dict]:
    """

    Args:
        independents: DataFrame
            Independent variables
        dependents: DataFrame
            Dependent variable
        metric_type: str in ['aic', 'bic', 'r2', 'r2_adj'
            'aic' refers Akaike information criterion
            'bic' refers Bayesian information criterion
            'r2' refers R-squared
            'r2_adj' refers Adjusted R-squared
        mutex_names: list
            Mutually exclusive variable names
            Like
            >>> _mutex_names = [('DEM_MEAN_100', 'DEM_MEAN_300')]
    Returns:
        independent name, dependent names, regression model, forward stepwise regression history
    """
    # preprocess
    assert isinstance(independents, DataFrame) and independents.shape[-1] == 1 and len(independents.shape) == 2
    independent_name = independents.columns.tolist()[0]
    independents = independents.values.reshape([-1])
    dependent_names = set(dependents.columns)
    selected_names = []
    best_metric = float('inf')
    history = dict()
    iterator_index = 0
    # forward stepwise regression
    while dependent_names:
        metric_with_variate = list()
        # get metric
        for new_name in dependent_names:
            model = get_model(independents, select_dependents(dependents, selected_names + [new_name]))
            metric = get_metric(model, metric_type)
            metric_with_variate.append((metric, new_name))
        metric_with_variate.sort(reverse=True)
        current_metric, new_name = metric_with_variate.pop()
        if current_metric < best_metric:
            # remove mutually exclusive variable names
            if mutex_names is None:
                dependent_names.remove(new_name)
            else:
                _mutex_names = [item for item in mutex_names if new_name in item]
                if len(_mutex_names) == 0:
                    dependent_names.remove(new_name)
                else:
                    for __mutex_names in _mutex_names:
                        for _mutex_name in __mutex_names:
                            if _mutex_name in dependent_names:
                                dependent_names.remove(_mutex_name)
            selected_names.append(new_name)
            best_metric = current_metric
        else:
            break
        history[iterator_index] = {
            'independent_name': independent_name,
            'dependent_names': selected_names,
            'metric': {
                metric_type: current_metric,
            }
        }
        iterator_index += 1
    return independent_name, selected_names, get_model(independents,
                                                       select_dependents(dependents, selected_names)), history
