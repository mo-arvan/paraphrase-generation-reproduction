import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from datetime import datetime


def plot_relative_preference():
    reproduction_results_df = pd.read_csv("results/lab1/tables/results.csv")
    hosking_results_df = pd.read_csv("results/original/results.csv")
    reproduction_results_df["From"] = "Ours"
    hosking_results_df["From"] = "Orig"

    stacked_df = pd.concat([reproduction_results_df, hosking_results_df], ignore_index=True)

    stacked_df["best_worst_scale"] = stacked_df["best_worst_scale"].apply(
        lambda x: round(x, 2))

    system_abbreviations_to_full = {
        "vae": "VAE",
        "lbow": "Latent BoW",
        "sep_ae": "Separator",
        "hrq": "HRQ-VAE",
    }

    stacked_df["system"] = stacked_df["system"].apply(
        lambda x: system_abbreviations_to_full[x])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax = sns.barplot(data=stacked_df,
                     x='From',
                     y='best_worst_scale',
                     hue='system',
                     palette="colorblind",
                     legend="brief"
                     )
    ax.set(xlabel='', ylabel="Relative Preference %")
    ax.legend_.set_title(None)
    plt.tight_layout()
    #
    for i in ax.containers:
        ax.bar_label(i, )

    plt.savefig("results/lab1/figures/reproduction_results.pdf")
    plt.savefig("results/lab1/figures/reproduction_results.png")
    plt.clf()


def plot_time_spent_on_pages():
    response_df = pd.read_csv("responses/responses.csv")

    def parse_time(time_str):
        date_part = time_str.split('GMT')[0].strip()
        return datetime.strptime(date_part, '%a %b %d %Y %H:%M:%S')

    steps = response_df["steps"].apply(lambda x: ast.literal_eval(x))

    welcome_page = steps.apply(lambda x: parse_time(x["welcome_page"]))
    instructions_page = steps.apply(lambda x: parse_time(x["instructions_page"]))
    task_page = steps.apply(lambda x: parse_time(x["task_page"]))
    finished_page = steps.apply(lambda x: parse_time(x["finished_page"]))

    time_spent_on_welcome = (instructions_page - welcome_page).dt.total_seconds()
    time_spent_on_instructions = (task_page - instructions_page).dt.total_seconds()
    time_spent_on_task = (finished_page - task_page).dt.total_seconds()

    time_spent_welcome_cap = time_spent_on_welcome.quantile(0.95)

    time_spent_on_welcome.loc[
        time_spent_on_welcome > time_spent_welcome_cap] = time_spent_welcome_cap

    time_spent_instructions_cap = time_spent_on_instructions.quantile(0.99)
    time_spent_on_instructions.loc[
        time_spent_on_instructions > time_spent_instructions_cap] = time_spent_instructions_cap

    time_spent_task_cap = time_spent_on_task.quantile(0.99)
    time_spent_on_task.loc[
        time_spent_on_task > time_spent_task_cap] = time_spent_task_cap

    # plot histogram of time spent on each page using sns.histplot
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    sns.histplot(time_spent_on_welcome, ax=axes[0], kde=True, stat='count', color='blue')
    sns.histplot(time_spent_on_instructions, ax=axes[1], kde=True, stat='count', color='blue')
    sns.histplot(time_spent_on_task, ax=axes[2], kde=True, stat='count', color='blue')

    axes[0].set_title(f"Welcome Page (cap: {time_spent_welcome_cap:.2f})")
    axes[1].set_title(f"Instructions Page (cap: {time_spent_instructions_cap:.2f})")
    axes[2].set_title(f"Task Page (cap: {time_spent_task_cap:.2f})")

    plt.tight_layout()
    plt.savefig("results/lab1/figures/time_spent_on_pages_hist.pdf")
    plt.savefig("results/lab1/figures/time_spent_on_pages_hist.png")

    plt.clf()

    time_df = pd.DataFrame({
        "Consent": time_spent_on_welcome,
        "Instructions": time_spent_on_instructions,
        "Task": time_spent_on_task
    })

    # plot boxplot of time spent on each page using sns.boxplot
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    ax = sns.boxplot(data=time_df)
    ax.set_title("Time spent on each page")
    plt.tight_layout()
    plt.savefig("results/lab1/figures/time_spent_on_pages_box.pdf")
    plt.savefig("results/lab1/figures/time_spent_on_pages_box.png")

    plt.clf()

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    sns.ecdfplot(time_spent_on_welcome, ax=axes[0], color='blue')
    sns.ecdfplot(time_spent_on_instructions, ax=axes[1], color='blue')
    sns.ecdfplot(time_spent_on_task, ax=axes[2], color='blue')

    axes[0].set_title(f"Welcome Page")
    axes[1].set_title(f"Instructions Page")
    axes[2].set_title(f"Task Page")

    axes[0].set_xlabel('Time spent (seconds)')
    axes[1].set_xlabel('Time spent (seconds)')
    axes[2].set_xlabel('Time spent (seconds)')

    # axes[0].set_ylabel('Cumulative Probability')
    # axes[1].set_ylabel('Cumulative Probability')
    # axes[2].set_ylabel('Cumulative Probability')

    plt.tight_layout()
    plt.savefig("results/lab1/figures/time_spent_on_pages_ecdf.pdf")
    plt.savefig("results/lab1/figures/time_spent_on_pages_ecdf.png")
    plt.clf()


def main():
    plot_time_spent_on_pages()

    plot_relative_preference()


if __name__ == '__main__':
    sns.set_theme(style="darkgrid")
    main()
