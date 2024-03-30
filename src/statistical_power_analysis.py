import csv

import statsmodels.stats.power as smp
from numpy import std, mean, sqrt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from analyze_human_eval import get_task_scores, load_and_preprocess_responses


def calculate_sample_size(groups, alpha, effect_size):
    # Define parameters
    power = 0.8  # Desired power level

    # Create an instance of the FTestAnovaPower class
    power_analysis = smp.FTestAnovaPower()

    # % We used the pwr library (Champely, 2020) in R (R
    # % Core Team, 2023) to run the following command:
    # % pwr.anova.test(k=5,f=.3,sig.level=.05,n=20)
    # % These numbers correspond to the number of different
    # % systems (5), desired effect size (0.3 or greater), significance

    # Calculate sample size
    sample_size = power_analysis.solve_power(effect_size=effect_size,
                                             nobs=None,
                                             alpha=alpha,
                                             power=power,
                                             k_groups=groups)

    print("Required Sample Size:", sample_size)

    return sample_size


def calculate_power(n_groups, alpha, effect_size, sample_size):
    # Define parameters

    # Create an instance of the FTestAnovaPower class
    power_analysis = smp.FTestAnovaPower()

    # Calculate power
    power = power_analysis.solve_power(effect_size=effect_size, nobs=sample_size, alpha=alpha, k_groups=n_groups)

    return power


def calculate_cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx - 1) * std(x, ddof=1) ** 2 + (ny - 1) * std(y, ddof=1) ** 2) / dof)


def calculate_effect_size(scores_df):
    # Initialize list to store Cohen's d for each pairwise comparison
    cohen_d_list = []

    reshaped_data = scores_df.melt(var_name='predictor', value_name='response')

    # Fit ANOVA model
    model = ols('response ~ C(predictor)', data=reshaped_data).fit()
    anova_table = anova_lm(model, typ=2)

    # Calculate partial eta-squared
    ss_between = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
    partial_eta_squared = ss_between / (len(anova_table) - 1)

    print("Partial eta-squared:")
    print(partial_eta_squared)

    anova_table.to_csv("results/lab1/tables/anova_table.csv", index=False, quoting=csv.QUOTE_NONNUMERIC,
                       float_format='%.2f')
    anova_table.to_latex("results/lab1/tables/anova_table.tex", index=False, float_format='%.2f', escape=True)
    effect_size, residual = partial_eta_squared.tolist()

    return effect_size, residual


def main():
    responses_processed_df = load_and_preprocess_responses()
    scores_df, system_count_dict = get_task_scores(responses_processed_df)

    # cohen j (1988) Statistical Power Analysis for Behavioral Sciences
    # Before looking at how to work out effect size, it might be worth looking at Cohen’s (1988) guidelines. According to him:
    #
    #     Small: 0.01
    #     Medium: 0.059
    #     Large: 0.138

    groups = 4
    desired_alpha = 0.05
    sample_size = len(scores_df)

    effect_size, residual = calculate_effect_size(scores_df)
    current_power = calculate_power(groups, desired_alpha, effect_size, sample_size)
    required_sample_size = calculate_sample_size(groups, desired_alpha, effect_size)

    adjusted_alpha = 0.052  # increase alpha to 0.1 to achieve desired power

    adjusted_power = calculate_power(groups, adjusted_alpha, effect_size, required_sample_size)

    with open("results/lab1/experiment_stats.txt", "w") as f:
        f.write(f"group_count: {groups}\n")
        f.write(f"desired_alpha: {desired_alpha}\n")
        f.write(f"sample_size: {sample_size}\n")
        f.write(f"effect_size: {effect_size}\n")
        f.write(f"residual: {residual}\n")
        f.write(f"current_power: {current_power}\n")
        f.write(f"required_sample_size: {required_sample_size}\n")
        f.write(f"adjusted_alpha: {adjusted_alpha}\n")
        f.write(f"adjusted_power: {adjusted_power}\n")


if __name__ == "__main__":
    main()
