"""Create dataset to be used for ASR pipelines for BizSpeech Dataset.

Author:
Anand C U
"""

from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import numpy as np
import pathlib
import json
import sys


def time_str_to_seconds(time_str):
    """
    Return time format in string to secods for easy airthmetic operations.

    Args : time_str : str with format hh:mm:ss
    """
    time_str = str(time_str)
    millisecs = 0
    if "." in time_str:
        time_str, millisecs = time_str.split(".")
    return int(time_str.split(":")[
        0]) * 3600 + int(time_str.split(":")[1]) * 60 + int(time_str.split(":")[2]) + float(millisecs) / 1000


class create_DataSet:
    """Class to create susets of Bizspeech dataset according to required native/nonnative speaker ratio, qna/presentation ratio."""

    def __init__(self, dataPath, totalDuration, nonnative, qna, strict_included, non_CEO_utt, seed, trainValTest, included_events=[], excluded_events=[]):
        """Init function.

        Args:
        ----
            - dataPath : Location of the dataset
            - totalDuration: total duration required for the subset. Used to calculate the sub totals according to ratios
            - included_events: Specify the eventIDs to choose from first before moving on to other data
            - excluded_events: Specify the eventIDs to exlude when creating subset. Might include invalid data, etc.
            - nonnative: Ratio of the dataset that needs to be nonnative. (1-nonnative) is the native percent.
            - qna: Ratio of the dataset that needs to be from qna section. (1-qna) is the presentation percent.
            - strict_included: Only pick events from included_events. Do not include others even if the data in included was not enough.
            - non_CEO_utt: Choose non CEO data from the files. Usually these do not have native/nonnative information.
            - seed: Seed to use for numpy randomisation

        """
        if non_CEO_utt:
            print("Native/ Non Native Criteria is not well defined for Non CEOs. The criteria ratios may not be accurate with this enabled")
        self.non_CEO_utt = non_CEO_utt
        self.dataset_dict = {}
        self.split_dicts = {"train": {}, "val": {}, "test": {}}
        self.dataPath = dataPath
        self.non_CEO_utt = non_CEO_utt
        self.bizspeech_metadata = self.load_Bizspeech_metadata()
        if not included_events:
            included_events = list(self.bizspeech_metadata.keys())
        event_list = list(set(included_events) - set(excluded_events))
        event_list = np.sort(event_list)
        np.random.seed(seed)
        np.random.shuffle(event_list)
        print("Picking data from ", len(event_list), " events.")
        totalDuration = totalDuration * 60 * 60 * 1000
        self.duration_dict_limit = {
            "nativepresentation": totalDuration * (1 - nonnative) * (1 - qna),
            "nonnativepresentation": totalDuration * nonnative * (1 - qna),
            "nativeq-and-a": totalDuration * (1 - nonnative) * qna,
            "nonnativeq-and-a": totalDuration * nonnative * qna
        }
        self.duration_dict_progress = {"nativepresentation": 0,
                                       "nonnativepresentation": 0,
                                       "nativeq-and-a": 0,
                                       "nonnativeq-and-a": 0
                                       }
        progress_complete = self.dataset_compile_from_event_list(event_list)
        if not progress_complete:
            if included_events:
                if strict_included:
                    print("Not all the categories were completed with the given criteria. Try freeing up the parameters and split ratios or reduce the number of hours of data.")
                else:
                    print(
                        "Using events from superset apart from included events to complete the required dataset.")
                    included_events_whole = list(
                        self.bizspeech_metadata.keys())
                    event_list = list(
                        (set(included_events_whole) - set(excluded_events)) - set(included_events))
                    event_list = np.sort(event_list)
                    np.random.seed(seed)
                    np.random.shuffle(event_list)
                    progress_complete = self.dataset_compile_from_event_list(
                        event_list)
                    if not progress_complete:
                        print(
                            "Superset was insufficient for the given data split ratios and number of hours. Try reducing them to get the required data split.")
            else:
                print("Not all the categories were completed with the given criteria. Try freeing up the parameters and split ratios or reduce the number of hours of data.")

        utterance_list = list(self.dataset_dict.keys())
        idxes_list = np.split(utterance_list, [int(trainValTest[0] * len(utterance_list)), int(
            (trainValTest[0] + trainValTest[1]) * len(utterance_list)), len(utterance_list)])
        for event in idxes_list[0]:
            self.split_dicts["train"][event] = self.dataset_dict[event]
        for event in idxes_list[1]:
            self.split_dicts["val"][event] = self.dataset_dict[event]
        for event in idxes_list[2]:
            self.split_dicts["test"][event] = self.dataset_dict[event]

    def load_Bizspeech_metadata(self):
        """Load Metadata from json file. Returns dictionary with eventID as key."""
        with open('metadata/BizSpeech_MetaData.json', 'r') as fh:
            result_dict = json.load(fh)
        return result_dict

    def parse_to_json(self, filepath):
        """Write the subset Dataset to a json file.

        'eventID-sentenceID' as key and start_time, end_time, text, speakerID, category as values.

        Args:
        ----
            - filepath : Location for the output JSONs.

        """
        with open(filepath + "train.json", "w") as fh:
            json.dump(self.split_dicts["train"], fh)
        with open(filepath + "val.json", "w") as fh:
            json.dump(self.split_dicts["val"], fh)
        with open(filepath + "test.json", "w") as fh:
            json.dump(self.split_dicts["test"], fh)

    def parse_to_csv(self, filepath):
        """Write the subset Dataset to a json file.

        'eventID-sentenceID', start_time, end_time, text, speakerID, category as the columns.

        Args:
        ----
            - filepath : Location of the output JSON.

        """
        import pandas as pd
        items = []
        for k in self.split_dicts["train"]:
            new_dict = {"utterance_ID": k}
            new_dict.update(self.dataset_dict[k])
            items.append(new_dict)
        pd.DataFrame(items).to_csv(filepath + "train.csv", index=False, columns=[
            "utterance_ID", "category", "start", "end", "spkID", "txt"])

        items = []
        for k in self.split_dicts["val"]:
            new_dict = {"utterance_ID": k}
            new_dict.update(self.dataset_dict[k])
            items.append(new_dict)
        pd.DataFrame(items).to_csv(filepath + "val.csv", index=False, columns=[
            "utterance_ID", "category", "start", "end", "spkID", "txt"])

        items = []
        for k in self.split_dicts["test"]:
            new_dict = {"utterance_ID": k}
            new_dict.update(self.dataset_dict[k])
            items.append(new_dict)
        pd.DataFrame(items).to_csv(filepath + "test.csv", index=False, columns=[
            "utterance_ID", "category", "start", "end", "spkID", "txt"])

    def dataset_compile_from_event_list(self, event_list):
        """Iterate over given event list and create the dataset dictionary based on data from TimeAligned Transcript.

        Args:
        ----
            - event_list: List of eventIDs that need to e processed.

        """
        for i, event in enumerate(event_list):
            event_metadata = self.bizspeech_metadata[event]
            if event_metadata["partition"] == 5:
                category = "native"
            else:
                category = "nonnative"
            ceo = event_metadata["ceoID"]
            transcript = np.genfromtxt(
                self.dataPath + event + "/transcript_timealigned.txt", names=True, dtype=None, delimiter="\t", encoding='utf-8')
            # print(transcript)
            for i, row in enumerate(transcript[:-1]):
                if row["SpeakerID"] == ceo or self.non_CEO_utt:
                    category_with_session = category + row["Session"]
                    if self.duration_dict_progress[category_with_session] < self.duration_dict_limit[category_with_session]:
                        sentence_ID = row["SentenceID"]
                        try:
                            utterance_ID = event + "-" + sentence_ID
                            utterance_end_time = transcript[i +
                                                            1]["SentenceTimeGen"]
                            duration = int((time_str_to_seconds(utterance_end_time) - time_str_to_seconds(
                                row["SentenceTimeGen"])) * 1000)
                            self.dataset_dict[utterance_ID] = {
                                "start": row["SentenceTimeGen"],
                                "end": utterance_end_time,
                                "spkID": row["SpeakerID"],
                                "txt": row["Sentence"],
                                "category": category_with_session
                            }
                            self.duration_dict_progress[category_with_session] += duration
                        except TypeError:
                            print(event, sentence_ID)
            if i % 100 == 0:
                print("Dataset Creation Progress")
                progress_complete = True
                for k in self.duration_dict_progress:
                    progress = self.duration_dict_progress[k] / \
                        self.duration_dict_limit[k] * 100
                    print(k, progress)
                    progress_complete *= progress > 100
                if progress_complete:
                    print(i)
                    break

        return progress_complete


def prepare_bizspeech_speechbrain(local_dataset_folder, data_folder, hours_reqd, nonnative=0.5, qna=0.5, strict_included=False, non_CEO_utt=False, seed=0, trainValTest=[0.6, 0.2], output_format="json", exclude_event_json=None, include_event_json=None):
    """Entrypoint for speechbrain function that uses a method directly as a parameter.

    Args:
    -----
        - local_dataset_folder: Folder to store the dataset files train, val and test JSONs
        - data_folder : Location of the dataset
        - hours_reqd: total duration required for the subset. Used to calculate the sub totals according to ratios
        - nonnative: Ratio of the dataset that needs to be nonnative. (1-nonnative) is the native percent.
        - qna: Ratio of the dataset that needs to be from qna section. (1-qna) is the presentation percent.
        - strict_included: Only pick events from included_events. Do not include others even if the data in included was not enough.
        - non_CEO_utt: Choose non CEO data from the files. Usually these do not have native/nonnative information.
        - seed: Seed to use for numpy randomisation
        - trainValTest: List/tuple of len 2. First is the train portion, second  is the val. Remaining will be the test portion.
        - output_format: Specify the output format. json/csv
        - included_events: Specify the eventIDs to choose from first before moving on to other data
        - excluded_events: Specify the eventIDs to exlude when creating subset. Might include invalid data, etc.

    """
    dest_dir = pathlib.Path(local_dataset_folder).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(exclude_event_json, bool(exclude_event_json))
    if exclude_event_json:
        sb.utils.data_utils.download_file(
            exclude_event_json, local_dataset_folder + exclude_event_json)
        with open(local_dataset_folder + exclude_event_json) as fh:
            exclude_list = [item for sublist in list(
                json.load(fh).values()) for item in sublist]
    if include_event_json:
        sb.utils.data_utils.download_file(
            include_event_json, local_dataset_folder + include_event_json)
        with open(local_dataset_folder + include_event_json) as fh:
            include_list = [item for sublist in list(
                json.load(fh).values()) for item in sublist]
    datasetObj = create_DataSet(data_folder, hours_reqd, nonnative, qna, strict_included, non_CEO_utt,
                                seed, trainValTest, included_events=include_list, excluded_events=exclude_list)
    if output_format == "json":
        datasetObj.parse_to_json(local_dataset_folder)
    else:
        datasetObj.parse_to_csv(local_dataset_folder)


if __name__ == '__main__':
    # Load hyper parameters file with command-line overrides
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.

    sb.utils.distributed.run_on_main(
        prepare_bizspeech_speechbrain,
        kwargs={
            "local_dataset_folder": hparams["local_dataset_folder"],
            "data_folder": hparams["data_folder"],
            "hours_reqd": hparams["hours_reqd"],
            "nonnative": hparams["nonnative"],
            "qna": hparams["qna"],
            "strict_included": hparams["strict_included"],
            "non_CEO_utt": hparams["non_CEO_utt"],
            "seed": hparams["seed"],
            "trainValTest": hparams["trainValTest"],
            "output_format": hparams["output_format"],
            "include_event_json": hparams["include_event_json"],
            "exclude_event_json": hparams["exclude_event_json"]
        },
    )
