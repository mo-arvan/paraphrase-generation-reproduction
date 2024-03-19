import csv

import krippendorff
import pandas as pd


def get_selected_systems(meaning_i):
    if meaning_i is False:
        value = 0
    elif meaning_i is True:
        value = 1
    else:
        raise ValueError(f"Unexpected value: {meaning_i}")
    return value


def process_responses_df(responses_df):
    results_list = []
    attention_check_list = ["distractor", "golds", "inputs"]

    for _, row in responses_df.iterrows():
        for i in range(32):
            selected_system = row[f"meaning{i}"]
            results_dict = dict()
            results_dict[f"systema"] = row[f"systema{i}"]
            results_dict[f"systemb"] = row[f"systemb{i}"]
            system_a_and_b = [results_dict[f"systema"], results_dict[f"systemb"]]

            if any([x in attention_check_list for x in system_a_and_b]):
                # it is not clear what to do when a participant fails the attention check
                continue

            results_dict["dataset"] = row[f"dataset{i}"]
            results_dict["dataset_index"] = row[f"ix{i}"]
            results_dict["dataset_id"] = f'{row[f"dataset{i}"]}-{row[f"ix{i}"]}'
            results_dict[f"selected_system"] = selected_system
            results_dict["input"] = row[f"input{i}"]
            results_dict["outputa"] = row[f"outputa{i}"]
            results_dict["outputb"] = row[f"outputb{i}"]
            results_dict["task_id"] = f'{row[f"dataset{i}"]}-{row[f"ix{i}"]}-{row[f"systema{i}"]}-{row[f"systemb{i}"]}'
            results_dict["participant_id"] = row["prolific_pid"]
            results_list.append(results_dict)

    results_df = pd.DataFrame(results_list)

    results_df["selected_system"] = results_df["selected_system"].apply(int)

    return results_df


def calculate_inter_annotator_agreement(responses_processed_df):
    annotation_matrix = (
        responses_processed_df[['task_id', "participant_id", "selected_system"]].pivot_table(
            index="participant_id",
            columns="task_id",
            values="selected_system",
            aggfunc="first")
        .reset_index(drop=True))

    annotation_matrix.to_csv("results/annotation_matrix.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    alpha = krippendorff.alpha(annotation_matrix.to_numpy(), level_of_measurement='nominal')

    with open("results/alpha.txt", "w") as f:
        f.write(f"Krippendorff's Alpha: {alpha:.3f}")
    print(f"Krippendorff's Alpha: {alpha:.3f}")
    return alpha


def print_datasets_used(responses_processed_df):
    datasets_and_index = responses_processed_df[["dataset", "dataset_index"]].drop_duplicates(keep="first")
    datasets_grouped = datasets_and_index[["dataset"]].groupby("dataset").agg(count=("dataset", "size")).reset_index()

    datasets_grouped.to_csv("results/datasets_used.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(datasets_grouped)


def calculate_metrics(responses_processed_df):
    metrics_list = []

    systems_set = set(responses_processed_df["systema"].tolist() + responses_processed_df["systemb"].tolist())

    for system in systems_set:
        system_a_matches = responses_processed_df[responses_processed_df["systema"] == system]
        system_b_matches = responses_processed_df[responses_processed_df["systemb"] == system]

        system_a_wins = len(system_a_matches[system_a_matches["selected_system"] == 0])
        system_a_losses = len(system_a_matches[system_a_matches["selected_system"] == 1])
        system_b_wins = len(system_b_matches[system_b_matches["selected_system"] == 1])
        system_b_losses = len(system_b_matches[system_b_matches["selected_system"] == 0])
        system_count = len(system_a_matches) + len(system_b_matches)

        wins_count = system_a_wins + system_b_wins
        losses_count = system_a_losses + system_b_losses

        total_count = wins_count + losses_count

        assert total_count == system_count

        best_worst_scale = None
        win_percentage = None

        if wins_count + losses_count != 0:
            best_worst_scale = (wins_count - losses_count) / total_count * 100.
            win_percentage = wins_count / total_count * 100.

        metrics_dict = {
            "system": system,
            "wins": wins_count,
            "losses": losses_count,
            "best_worst_scale": best_worst_scale,
            "best_worst_score": wins_count - losses_count,
            "win_percentage": win_percentage,
        }
        metrics_list.append(metrics_dict)

    metrics_df = pd.DataFrame(metrics_list)

    metrics_df.to_csv("results/metrics.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(metrics_df)

    return metrics_df


def main():
    responses_df = pd.read_csv('responses/responses.csv')
    responses_processed_df = process_responses_df(responses_df)

    print_datasets_used(responses_processed_df)

    metrics_df = calculate_metrics(responses_processed_df)

    alpha = calculate_inter_annotator_agreement(responses_processed_df)


if __name__ == "__main__":
    main()
