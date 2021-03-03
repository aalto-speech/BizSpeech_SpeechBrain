"""Create dataset to be used for ASR pipelines for BizSpeech Dataset.

Author:
Anand C U
"""

import json

import numpy as np


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

    def __init__(self, dataPath, totalDuration, included_events=None, excluded_events=[], nonnative=0.5, qna=0.5, strict_included=False, non_CEO_utt=False):
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

        """
        if non_CEO_utt:
            print("Native/ Non Native Criteria is not well defined for Non CEOs. The criteria ratios may not be accurate with this enabled")
        self.non_CEO_utt = non_CEO_utt
        self.dataset_dict = {}
        self.dataPath = dataPath
        self.non_CEO_utt = non_CEO_utt
        self.bizspeech_metadata = self.load_Bizspeech_metadata()
        if not included_events:
            included_events = list(self.bizspeech_metadata.keys())
        event_list = list(set(included_events) - set(excluded_events))
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
            if strict_included:
                print("Not all the categories were completed with the given criteria. Try freeing up the parameters and split ratios or reduce the number of hours of data.")
            else:
                print(
                    "Using events from superset apart from included events to complete the required dataset.")
                included_events = list(self.bizspeech_metadata.keys())
                event_list = list(set(included_events) - set(excluded_events))
                progress_complete = self.dataset_compile_from_event_list(
                    event_list)
                if not progress_complete:
                    print("Superset was insufficient for the given data split ratios and number of hours. Try reducing them to get the required data split.")

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
            - filepath : Location of the output JSON.

        """
        with open(filepath, "w") as fh:
            json.dump(self.dataset_dict, fh)

    def parse_to_csv(self, filepath):
        """Write the subset Dataset to a json file.

        'eventID-sentenceID', start_time, end_time, text, speakerID, category as the columns.

        Args:
        ----
            - filepath : Location of the output JSON.

        """
        import pandas as pd
        items = []
        for k in self.dataset_dict:
            new_dict = {"utterance_ID": k}
            new_dict.update(self.dataset_dict[k])
            items.append(new_dict)
        pd.DataFrame(items).to_csv(filepath, index=False, columns=[
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


with open("metadata/exclude_event_list.json") as fh:
    exclude_list = [item for sublist in list(
        json.load(fh).values()) for item in sublist]
with open("metadata/cloudasr_testset_lists.json") as fh:
    include_list = [item for sublist in list(
        json.load(fh).values()) for item in sublist]
datasetObj = create_DataSet("/m/triton/scratch/biz/bizspeech/MEDIA/",
                            200, included_events=include_list, excluded_events=exclude_list)
datasetObj.parse_to_json("datasets/dataset.json")
# datasetObj.parse_to_csv("datasets/dataset.csv")
