"""Create dataset to be used for ASR pipelines for BizSpeech Dataset.

Author:
Anand C U
"""

import json
import logging
import pathlib
import shutil
import sys

import numpy as np
import speechbrain as sb
import torchaudio
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


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

    def __init__(self, hparams: dict, included_events: list = [], excluded_events: list = []):
        """Init function.

        Args:
            hparams (dict): Hyperparameters dictionary, usually contains contents from 'dataset.yaml' loaded to a dict. 
            included_events (list, optional): Specify the eventIDs to choose from first before moving on to other data. Defaults to [].
            excluded_events (list, optional): Specify the eventIDs to exlude when creating subset. Might include invalid data, etc.. Defaults to [].
        """

        if hparams["non_CEO_utt"]:
            logger.info(
                "Native/ Non Native Criteria is not well defined for Non CEOs. The criteria ratios may not be accurate with this enabled")
        self.dataset_dict = {}
        self.split_dicts = {"train": {}, "val": {}, "test": {}}
        self.dataPath = hparams["data_folder"]
        self.non_CEO_utt = hparams["non_CEO_utt"]
        self.bizspeech_metadata = self.load_Bizspeech_metadata()
        self.utterance_duration_limit = hparams["utterance_duration_limit"]
        self.audio_filetype = hparams["audio_filetype"]
        trainValTest = hparams["trainValTest"]
        if not included_events:
            included_events = list(self.bizspeech_metadata.keys())
        event_list = list(set(included_events) - set(excluded_events))
        event_list = np.sort(event_list)
        np.random.seed(hparams["seed"])
        np.random.shuffle(event_list)
        logger.info(f"Picking data from {len(event_list)} events.")
        totalDuration = hparams["hours_reqd"] * 60 * 60 * 1000
        nonnative = hparams["nonnative"]
        qna = hparams["qna"]
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
                if hparams["strict_included"]:
                    logger.info(
                        "Not all the categories were completed with the given criteria. Try freeing up the parameters and split ratios or reduce the number of hours of data.")
                else:
                    logger.info(
                        "Using events from superset apart from included events to complete the required dataset.")
                    included_events_whole = list(
                        self.bizspeech_metadata.keys())
                    event_list = list(
                        (set(included_events_whole) - set(excluded_events)) - set(included_events))
                    event_list = np.sort(event_list)
                    np.random.seed(hparams["seed"])
                    np.random.shuffle(event_list)
                    progress_complete = self.dataset_compile_from_event_list(
                        event_list)
                    if not progress_complete:
                        logger.info(
                            "Superset was insufficient for the given data split ratios and number of hours. Try reducing them to get the required data split.")
            else:
                logger.info(
                    "Not all the categories were completed with the given criteria. Try freeing up the parameters and split ratios or reduce the number of hours of data.")

        utterance_list = list(self.dataset_dict.keys())
        idxes_list = np.split(utterance_list, [int(trainValTest[0] * len(utterance_list)), int(
            (trainValTest[0] + trainValTest[1]) * len(utterance_list)), len(utterance_list)])
        for event in idxes_list[0]:
            self.split_dicts["train"][event] = self.dataset_dict[event]
        for event in idxes_list[1]:
            self.split_dicts["val"][event] = self.dataset_dict[event]
        for event in idxes_list[2]:
            self.split_dicts["test"][event] = self.dataset_dict[event]

        if hparams["sorting"] == "ascending":
            self.split_dicts["val"] = {k: self.split_dicts["val"][k] for k in sorted(
                self.split_dicts["val"].keys(), key=lambda x: self.split_dicts["val"][x]["duration"])}
            self.split_dicts["train"] = {k: self.split_dicts["train"][k] for k in sorted(
                self.split_dicts["train"].keys(), key=lambda x: self.split_dicts["train"][x]["duration"])}
            self.split_dicts["test"] = {k: self.split_dicts["test"][k] for k in sorted(
                self.split_dicts["test"].keys(), key=lambda x: self.split_dicts["test"][x]["duration"])}
        elif hparams["sorting"] == "descending":
            reverse = True
            self.split_dicts["val"] = {k: self.split_dicts["val"][k] for k in sorted(
                self.split_dicts["val"].keys(), key=lambda x: self.split_dicts["val"][x]["duration"], reverse=reverse)}
            self.split_dicts["train"] = {k: self.split_dicts["train"][k] for k in sorted(
                self.split_dicts["train"].keys(), key=lambda x: self.split_dicts["train"][x]["duration"], reverse=reverse)}
            self.split_dicts["test"] = {k: self.split_dicts["test"][k] for k in sorted(
                self.split_dicts["test"].keys(), key=lambda x: self.split_dicts["test"][x]["duration"], reverse=reverse)}
        else:
            # No sorting required
            pass

    def load_audio(self, audio: str, start: str, end: str, copy_to_local: bool, local_dest_dir: str, audio_filetype: str, preprocess_audio):
        """Load the audio signal.

        Args:
            audio (str): Filepath of the audio file
            start (str): Start timestamp of the audio utterance
            end (str): End timestamp of the audio utterance
            copy_to_local (bool): [description]
            local_dest_dir (str): [description]
            audio_filetype (str): [description]
            preprocess_audio ([type]): [description]

        Returns:
            sig: Audio signal cropped and preprocessed.
        """

        if copy_to_local:
            event_id = str(pathlib.Path(audio).resolve().parent).split("/")[-1]
            dest_dir = pathlib.Path(
                local_dest_dir).resolve().joinpath(event_id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_dir = dest_dir.joinpath("recording." + audio_filetype)
            if not dest_dir.is_file():
                audio = shutil.copy(audio, dest_dir)

        metadata = torchaudio.info(audio)
        start = int(time_str_to_seconds(start) * metadata.sample_rate)
        end = int(time_str_to_seconds(end) * metadata.sample_rate)
        num_frames = end - start
        audio, fs = torchaudio.load(
            audio, num_frames=num_frames, frame_offset=start)
        audio = audio.transpose(0, 1)
        sig = preprocess_audio(audio, metadata.sample_rate)
        return sig

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
        with open(filepath + "/train.json", "w") as fh:
            json.dump(self.split_dicts["train"], fh)
        with open(filepath + "/val.json", "w") as fh:
            json.dump(self.split_dicts["val"], fh)
        with open(filepath + "/test.json", "w") as fh:
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
        pd.DataFrame(items).to_csv(filepath + "/train.csv", index=False, columns=[
            "utterance_ID", "category", "start", "end", "spkID", "txt"])

        items = []
        for k in self.split_dicts["val"]:
            new_dict = {"utterance_ID": k}
            new_dict.update(self.dataset_dict[k])
            items.append(new_dict)
        pd.DataFrame(items).to_csv(filepath + "/val.csv", index=False, columns=[
            "utterance_ID", "category", "start", "end", "spkID", "txt"])

        items = []
        for k in self.split_dicts["test"]:
            new_dict = {"utterance_ID": k}
            new_dict.update(self.dataset_dict[k])
            items.append(new_dict)
        pd.DataFrame(items).to_csv(filepath + "/test.csv", index=False, columns=[
            "utterance_ID", "category", "start", "end", "spkID", "txt"])

    def json_to_webdataset_tar(self, dataset: str, dataset_dir: str, shard_maxcount: int, copy_to_local: bool, local_dest_dir: str, audio_filetype: str, preprocess_audio):
        """Convert dict input to webdataset tar format.

        Args:
            dataset_dict (dict): Dictionary with the dataset parameters 
            hparams (dict): Hyperparameters dictionary loaded from YAML
        """
        dataset_dict = self.split_dicts[dataset]

        pattern = str(dataset_dir + "/" + dataset + "_shard" + "-%06d.tar")
        with wds.ShardWriter(pattern, maxcount=shard_maxcount) as sink:
            for audio_utt in tqdm(dataset_dict):
                sig = self.load_audio(audio=dataset_dict[audio_utt]["audio"], start=dataset_dict[audio_utt]["start"], end=dataset_dict[audio_utt]["end"],
                                      copy_to_local=copy_to_local, local_dest_dir=local_dest_dir, audio_filetype=audio_filetype, preprocess_audio=preprocess_audio)
                sample = {
                    "__key__": audio_utt,
                    "wav.pyd": sig,
                    "meta.json": {
                        "speaker_id": dataset_dict[audio_utt]["spkID"],
                        "category": dataset_dict[audio_utt]["category"],
                        "utterance_id": audio_utt,
                        "txt": dataset_dict[audio_utt]["txt"],
                    },
                }
                sink.write(sample)

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
                self.dataPath + "/" + event + "/transcript_timealigned.txt", names=True, dtype=None, delimiter="\t", encoding='utf-8')
            for i, row in enumerate(transcript[:-1]):
                if row["SpeakerID"] == ceo or self.non_CEO_utt:
                    category_with_session = category + row["Session"]
                    if self.duration_dict_progress[category_with_session] < self.duration_dict_limit[category_with_session]:
                        sentence_ID = row["SentenceID"]
                        utterance_ID = event + "-" + sentence_ID
                        utterance_end_time = transcript[i +
                                                        1]["SentenceTimeGen"]
                        end_time_in_secs = time_str_to_seconds(
                            utterance_end_time)
                        start_time_in_secs = time_str_to_seconds(
                            row["SentenceTimeGen"])
                        duration = int(
                            (end_time_in_secs - start_time_in_secs) * 1000)
                        if int(duration / 1000) < self.utterance_duration_limit:
                            self.dataset_dict[utterance_ID] = {
                                "audio": self.dataPath + "/" + event + "/recording." + self.audio_filetype,
                                "start": row["SentenceTimeGen"],
                                "end": utterance_end_time,
                                "duration": duration,
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
                    logger.info(f"Finished compiling dataset from {i} events.")
                    break

        return progress_complete


def prepare_bizspeech_speechbrain(hparams: dict):
    """Entrypoint for speechbrain function that uses a method directly as a parameter.

    Args:
        hparams ([type]): [description]
    """
    training_file = pathlib.Path(
        hparams["local_dataset_folder"] + "/train.json")
    if training_file.is_file():
        logger.info(f"{training_file} exists. Skipping dataset preparation.")
        return
    dest_dir = pathlib.Path(hparams["local_dataset_folder"]).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    if hparams["exclude_event_json"]:
        sb.utils.data_utils.download_file(
            hparams["exclude_event_json"], hparams["local_dataset_folder"] + "/exclude_list.json")
        with open(hparams["local_dataset_folder"] + "/exclude_list.json") as fh:
            exclude_list = [item for sublist in list(
                json.load(fh).values()) for item in sublist]
    if hparams["include_event_json"]:
        sb.utils.data_utils.download_file(
            hparams["include_event_json"], hparams["local_dataset_folder"] + "/include_list.json")
        with open(hparams["local_dataset_folder"] + "/include_list.json") as fh:
            include_list = [item for sublist in list(
                json.load(fh).values()) for item in sublist]
    datasetObj = create_DataSet(
        hparams, included_events=include_list, excluded_events=exclude_list)
    if hparams["output_format"] == "json":
        datasetObj.parse_to_json(hparams["local_dataset_folder"])
        logger.info(
            f'Created JSON dataset files under {hparams["local_dataset_folder"]} directory.')
        if hparams["use_wds"]:
            datasetObj.json_to_webdataset_tar("train", hparams["local_dataset_folder"], hparams["shard_maxcount"],
                                              hparams["copy_to_local"], hparams["local_dest_dir"], hparams["audio_filetype"], hparams["preprocess_audio"])
            datasetObj.json_to_webdataset_tar("val", hparams["local_dataset_folder"], hparams["shard_maxcount"],
                                              hparams["copy_to_local"], hparams["local_dest_dir"], hparams["audio_filetype"], hparams["preprocess_audio"])
            datasetObj.json_to_webdataset_tar("test", hparams["local_dataset_folder"], hparams["shard_maxcount"],
                                              hparams["copy_to_local"], hparams["local_dest_dir"], hparams["audio_filetype"], hparams["preprocess_audio"])
    else:
        datasetObj.parse_to_csv(hparams["local_dataset_folder"])
        if hparams["use_wds"]:
            logger.info("Webdataset from csv is not supported!")


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
            "hparams": hparams,
        },
    )
