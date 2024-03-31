from scipy.stats import pearsonr, spearmanr
import cv
import pandas as pd
import csv


def calculate_pearson_spearman_correlation(set_a, set_b):
    # Pearson's r
    pearson_corr, pearson_p = pearsonr(set_a, set_b)

    # Spearman's ρ
    spearman_corr, spearman_p = spearmanr(set_a, set_b)
    #
    # print("Pearson's r: ", pearson_corr, "p-value: ", pearson_p)
    # print("Spearman's ρ: ", spearman_corr, "p-value: ", spearman_p)

    pearson_result_dict = {
        "input a": set_a,
        "input b": set_b,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
    }
    spearman_result_dict = {
        "input a": set_a,
        "input b": set_b,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
    }
    df = pd.DataFrame([pearson_result_dict])
    df.to_csv("results/lab1/tables/pearson.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, )
    df.to_latex("results/lab1/tables/pearson.tex", float_format="{:0.2f}".format, escape=True)

    print(df)
    df = pd.DataFrame([spearman_result_dict])
    df.to_csv("results/lab1/tables/spearman.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, )
    df.to_latex("results/lab1/tables/spearman.tex", float_format="{:0.2f}".format, escape=True)
    print(df)


def calculate_coefficient_of_variation(set_a, set_b, range_start, range_end):
    if range_start < 0:
        adjusted_set_a = set_a + abs(range_start)
        adjusted_set_b = set_b + abs(range_start)
    else:
        adjusted_set_a = set_a
        adjusted_set_b = set_b

    full_result_list = []
    for result_a, result_b in zip(adjusted_set_a, adjusted_set_b):
        values = [result_a, result_b]

        precision_results = cv.get_precision_results(values)

        full_result_list.append(precision_results)
    df = pd.DataFrame(full_result_list)

    df.to_latex("results/lab1/tables/cv_2_way.tex", float_format="{:0.2f}".format, escape=True)
    print(df)


def main():
    results_df = pd.read_csv("results/lab1/tables/results.csv")
    hosking_results = pd.read_csv("results/original/results.csv")

    results_df.sort_values(by="system", inplace=True)
    hosking_results.sort_values(by="system", inplace=True)
    # need to ensure that the data is in the same order, this can be done using the system column
    original_values = hosking_results["best_worst_scale"].values
    ours_values = results_df["best_worst_scale"].values

    calculate_pearson_spearman_correlation(original_values, ours_values)
    calculate_coefficient_of_variation(original_values, ours_values, -100, 100)


if __name__ == "__main__":
    main()
