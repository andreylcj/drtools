""" 
This module was created to realize
Statistical Tests in order to quantify
distributions relations.

"""


from typing import Tuple, List
from scipy.stats import ttest_ind as stats_ttest_ind
from scipy.stats import chi2_contingency as stats_chi2_contingency
from scipy.stats import mannwhitneyu as stats_mannwhitneyu
from pandas import DataFrame, crosstab


def mann_whitney_u_test(
    dataframe: DataFrame,
    numerical_variable: str,
    categorical_variable: str,
) -> Tuple[float, float]:
    """Perform the Mann-Whitney U rank test between 1 numerical and 1 categorical column 
    with only two distinct classes..
    
    Perform the Mann-Whitney U rank test on two independent samples. The 
    Mann-Whitney U test is a nonparametric test of the 
    null hypothesis that the distribution underlying sample 
    x is the same as the distribution underlying 
    sample y. It is often used as a test of difference 
    in location between distributions.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame
    numerical_variable : str
        Name of numerical column for the Mann-Whitney U test.
    categorical_variable : str
        Name of categorical column for the Mann-Whitney U test.

    Returns
    -------
    Tuple[float, float]
        The values of Test Statistic and the p-value.
    """
    
    df = dataframe.loc[:, [numerical_variable, categorical_variable]]
    df = df.dropna()    
    categorical_values = list(
        df[categorical_variable] \
        .value_counts() \
        .index
    )
    
    assert len(categorical_values) == 2, \
        "Categorical column must have only 2 distinct classes."
        
    statistic, p_value = stats_ttest_ind(
        df[
            df[categorical_variable] == categorical_values[0]
        ][numerical_variable], 
        df[
            df[categorical_variable] == categorical_values[1]
        ][numerical_variable]
    )
    statistic, p_value = stats_mannwhitneyu(
        df[numerical_variable],
        df[categorical_variable],
    )
    return statistic, p_value


def t_test_ind(
    dataframe: DataFrame,
    numerical_variable: str,
    categorical_variable: str,
) -> Tuple[float, float]:
    """Perform the Two-Sample T-Test between 1 numerical and 1 categorical column 
    with only two distinct classes.
    
    Calculate the T-test for the means of two independent samples of scores.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame
    numerical_variable : str
        Name of numerical column for the T-Test
    categorical_variable : str
        Name of categorical column for the T-Test

    Returns
    -------
    Tuple[float, float]
        The values of Test Statistic and the p-value.
    """
    temp_df = dataframe[[numerical_variable, categorical_variable]]
    temp_df = temp_df.dropna()    
    categorical_values = list(
        temp_df[categorical_variable] \
        .value_counts() \
        .index
    )
    
    assert len(categorical_values) == 2, \
        "Categorical column must have only 2 distinct classes."
        
    t_stat, p = stats_ttest_ind(
        temp_df[
            temp_df[categorical_variable] == categorical_values[0]
        ][numerical_variable], 
        temp_df[
            temp_df[categorical_variable] == categorical_values[1]
        ][numerical_variable]
    )
    return t_stat, p


def chi2_contingency(
    dataframe: DataFrame,
    categorical_column1: str,
    categorical_column2: str
) -> Tuple[float, float, int, List[List[float]]]:
    """Realize Chi-Squared contigency test between 2 categorical 
    columns of DataFrame.
    
    Chi-square test of independence of variables in a contingency table.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame
    categorical_column1 : str
        Name of first categorical column
    categorical_column2 : str
        Name of second categorical column

    Returns
    -------
    Tuple[float, float, int, List[List[float]]]
        The Chi-squared Stat value, the p-value, the degrees of freedom
        and the expected matrix.
    """
    
    df = dataframe.copy()
    all_categories_col1 = list(df[categorical_column1].value_counts().index)
    all_categories_col2 = list(df[categorical_column2].value_counts().index)
    cross_tab = crosstab(df[categorical_column1], df[categorical_column2], margins=True)
    observed = cross_tab.iloc[0:-1, 0:-1]
    observed.columns = all_categories_col2
    observed.index = all_categories_col1
    chi_squared_stat, p_value, degree_of_freedom, expected \
        = stats_chi2_contingency(observed=observed)
    return chi_squared_stat, p_value, degree_of_freedom, expected