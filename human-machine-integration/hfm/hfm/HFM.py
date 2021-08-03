import numpy as np
import statsmodels.formula.api as sm


def step_aic(independent_variables, dependent_variables, **kwargs):
    """
    This function is used to select the best formula with smallest AIC
    Both exog and endog values can be either str or list.

    Note: This adopt only "forward" selection

    Args:
        independent_variables (list): independent variables
        dependent_variables (list): dependent variables
        kwargs: extra keyword argments for model (e.g., data, family)

    Returns:
        model: a model that seems to have the smallest AIC
        selected: a list of the columns selected as best field
    """
    model = sm.ols
    independent_variables = np.r_[[independent_variables]].flatten()
    dependent_variables = np.r_[[dependent_variables]].flatten()
    remaining = set(independent_variables)
    selected = []

    formula_head = ' + '.join(dependent_variables) + ' ~ '
    formula = formula_head + '1'
    aic = model(formula=formula, **kwargs).fit().aic
    print('AIC: {}, formula: {}'.format(round(aic, 3), formula))

    current_score, best_new_score = np.ones(2) * aic

    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula_tail = ' + '.join(selected + [candidate])
            formula = formula_head + formula_tail
            aic = model(formula=formula, **kwargs).fit().aic
            # print('AIC: {}, formula: {}'.format(round(aic, 3), formula))

            scores_with_candidates.append((aic, candidate))

        scores_with_candidates.sort()
        scores_with_candidates.reverse()
        best_new_score, best_candidate = scores_with_candidates.pop()

        if best_new_score < current_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score

    formula = formula_head + ' + '.join(selected)
    # print('The best formula: {}'.format(formula))
    return model(formula, **kwargs).fit(), selected