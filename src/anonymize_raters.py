import ast
import csv

import pandas as pd

worker_id_to_anonymized_id = {}


def generate_anonymized_id(worker_id):
    global worker_id_to_anonymized_id
    if worker_id in worker_id_to_anonymized_id:
        return worker_id_to_anonymized_id[worker_id]

    random_string = f'anon_worker_{len(worker_id_to_anonymized_id)}'
    worker_id_to_anonymized_id[worker_id] = random_string
    return random_string


def parse_json_str(json_str):
    response_dict = ast.literal_eval(json_str)
    response_dict["clicks"] = ast.literal_eval(response_dict["clicks"])
    response_dict["steps"] = ast.literal_eval(response_dict["steps"])
    response_dict["prolific_pid"] = generate_anonymized_id(response_dict["prolific_pid"])

    return response_dict


def get_selected_systems(meaning_i):
    if meaning_i is False:
        value = 0
    elif meaning_i is True:
        value = 1
    else:
        raise ValueError(f"Unexpected value: {meaning_i}")

    return value


def main():
    tasks_joined_df = pd.read_csv('responses/tasks_joined.csv')
    # tasks_results_df = pd.read_csv('responses/task_results.csv')

    tasks_joined_df["response_dict"] = tasks_joined_df["json_string"].apply(parse_json_str)

    tasks_joined_df.drop(columns=["json_string"], inplace=True)

    response_columns = list(tasks_joined_df["response_dict"][0].keys())

    for column in response_columns:
        tasks_joined_df[column] = tasks_joined_df["response_dict"].apply(lambda x: x[column])

    tasks_joined_df.to_csv("responses/responses.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    main()
