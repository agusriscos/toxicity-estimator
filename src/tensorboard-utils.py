import os
from os.path import join
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tensorboard_to_pandas(path: str) -> pd.DataFrame:
    default_size_guidance = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    event_acc = EventAccumulator(path, default_size_guidance)
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]

    for tag in tags:
        event_list = event_acc.Scalars(tag)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        r = {"metric": [tag] * len(step), "value": values, "step": step}
        r = pd.DataFrame(r)
        runlog_data = pd.concat([runlog_data, r])

    return runlog_data


def many_tensorboard_to_pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tensorboard_to_pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def tensorboard_event_to_dataframe(dataframe_name: str, logdir_or_logfile: str, out_dir: str):
    if os.path.isdir(logdir_or_logfile):
        event_paths = glob.glob(os.path.join(logdir_or_logfile, "event*"))
    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    if event_paths:
        print("Found tensorflow logs to process:")
        print(event_paths)
        all_logs = many_tensorboard_to_pandas(event_paths)

        os.makedirs(out_dir, exist_ok=True)
        print("Saving to csv file")
        out_file = os.path.join(out_dir, dataframe_name)
        print(out_file)
        all_logs.to_csv(out_file, index=None)
    else:
        print("No event paths have been found.")


if __name__ == "__main__":
    output_directory_path = '../training/bert'
    for df_name, log_dirpath in zip(["train-metrics.csv", "test-metrics.csv"],
                                    [join(output_directory_path, ""),
                                     join(output_directory_path, "")]):
        tensorboard_event_to_dataframe(
            dataframe_name=df_name,
            logdir_or_logfile=log_dirpath,
            out_dir=output_directory_path
        )
