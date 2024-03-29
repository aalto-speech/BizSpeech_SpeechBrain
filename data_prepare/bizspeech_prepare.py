"""Create dataset to be used for ASR pipelines for BizSpeech Dataset.

Author:
Anand C U
"""

import json
import logging
import pathlib
import shutil
import sys
import multiprocessing
#from multiprocessing import Process

import numpy as np
import speechbrain as sb
import torchaudio
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

logger = logging.getLogger(__name__)
the_queue = multiprocessing.Queue()

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
        np.random.seed(hparams["dataset_seed"])
        np.random.shuffle(event_list)
        logger.info(f"Picking data from {len(event_list)} events.")
        totalDuration = hparams["hours_reqd"] * 60 * 60 * 1000
        nonnative = hparams["nonnative"]
        qna = hparams["qna"]
        '''
        self.duration_dict_limit = {
            "nativepresentation": totalDuration * (1 - nonnative) * (1 - qna),
            "nonnativepresentation": totalDuration * nonnative * (1 - qna),
            "nativeq-and-a": totalDuration * (1 - nonnative) * qna,
            "nonnativeq-and-a": totalDuration * nonnative * qna
        }'''
        self.duration_dict_limit = {
            "nativepresentation": totalDuration,
            "nonnativepresentation": totalDuration,
            "nativeq-and-a": totalDuration,
            "nonnativeq-and-a": totalDuration
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
                    np.random.seed(hparams["dataset_seed"])
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

    def load_audio(self, audio: str, start: str, end: str, preprocess_audio):
        """Load the audio signal.

        Args:
            audio (str): Filepath of the audio file
            start (str): Start timestamp of the audio utterance
            end (str): End timestamp of the audio utterance
            preprocess_audio ([type]): [description]

        Returns:
            sig: Audio signal cropped and preprocessed.
        """

        '''
        if copy_to_local:
            event_id = str(pathlib.Path(audio).resolve().parent).split("/")[-1]
            dest_dir = pathlib.Path(
                local_dest_dir).resolve().joinpath(event_id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_dir = dest_dir.joinpath("recording." + audio_filetype)
            if not dest_dir.is_file():
                audio = shutil.copy(audio, dest_dir)
        '''
        metadata = torchaudio.info(audio)
        start = int(time_str_to_seconds(start) * metadata.sample_rate)
        end = int(time_str_to_seconds(end) * metadata.sample_rate)
        num_frames = end - start
        audio, fs = torchaudio.load(
            audio, num_frames=num_frames, frame_offset=start)
        audio = audio.transpose(0, 1)
        sig = preprocess_audio(audio, metadata.sample_rate)
        return sig

    def json_to_webdataset_tar(self, dataset: str, dataset_dir: str, number_of_shards: int, use_compression:bool, preprocess_audio):
        """Convert dict input to webdataset tar format.

        Args:
            dataset_dict (dict): Dictionary with the dataset parameters 
            hparams (dict): Hyperparameters dictionary loaded from YAML
        """
        def writeTar(queue):
            while True:
                fname, dataset_dict = queue.get(True)
                if fname is None:
                    break
                logger.info("Writing " + str(fname) + ".")
                with wds.TarWriter(fname, compress=use_compression) as sink:
                    for audio_utt in dataset_dict:
                        sig = self.load_audio(audio=dataset_dict[audio_utt]["audio"], start=dataset_dict[audio_utt]
                                                ["start"], end=dataset_dict[audio_utt]["end"], preprocess_audio=preprocess_audio)
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

        dataset_dict = self.split_dicts[dataset]
        if use_compression:
            pattern = str(dataset_dir + "/" + "bizspeech_shard" + "-%06d.tar.gz")
        else:
            pattern = str(dataset_dir + "/" + dataset + "_shard" + "-%06d.tar")

        if dataset == "train":
            shard_name_list = [pattern % i for i in range(number_of_shards)]
            per_shard = int(len(dataset_dict)/number_of_shards)
            split_indices = [
                (i+1)*per_shard for i in range(number_of_shards-1)] + [len(dataset_dict)]
            dataset_keys_splits = [list(a) for a in np.split(
                np.array(list(dataset_dict.keys())), split_indices)]
            dataset_values_splits = [[dataset_dict[key] for key in dataset_keys_split]
                                     for dataset_keys_split in dataset_keys_splits]
            dataset_dict_splits = [{dataset_keys_splits[i][j]: dataset_values_splits[i][j] for j in range(
                len(dataset_keys_splits[i]))} for i in range(len(dataset_keys_splits))]
            #p = Pool(cpu_count() - 1)
            #print(tuple(zip(shard_name_list, dataset_dict_splits)))
            #r = list(p.imap(writeTar, tuple(zip(shard_name_list, dataset_dict_splits))))
            # r.join()
            logger.info("The node has " + str(multiprocessing.cpu_count()) + "cores.")
            the_pool = multiprocessing.Pool(multiprocessing.cpu_count()-1, writeTar,(the_queue,))
            #jobs = []
            for i in range(number_of_shards):
                the_queue.put((shard_name_list[i], dataset_dict_splits[i]))
            for i in range(multiprocessing.cpu_count()-1):
                the_queue.put((None, None))
                #jobs.append(Process(target=writeTar, args=(
                #    shard_name_list[i], dataset_dict_splits[i])))
            #for j in jobs:
            #    j.start()
            #for j in jobs:
            #    j.join()
            # prevent adding anything more to the queue and wait for queue to empty
            the_queue.close()
            the_queue.join_thread()

            # prevent adding anything more to the process pool and wait for all processes to finish
            the_pool.close()
            the_pool.join()
            print("Completed tar generation")
            #Parallel(n_jobs=number_of_shards)(delayed(writeTar)(shard_name_list[i], dataset_dict_splits[i]) for i in range(number_of_shards))
        else:
            writeTar(pattern % 0, dataset_dict)

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
                        if row["Sentence"] != "":
                            try:
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
                            except:
                                continue
            if i % 100 == 0:
                print("Dataset Creation Progress")
                progress_complete = True
                total_duration = 0
                for k in self.duration_dict_progress:
                    progress = self.duration_dict_progress[k] / \
                        self.duration_dict_limit[k] * 100
                    total_duration += self.duration_dict_progress[k]
                    print(k, progress)
                    progress_complete *= progress > 100
                if progress_complete:
                    logger.info(f"Finished compiling dataset from {i} events.")
                    logger.info(f"Total duration in the dataset is {total_duration/3600000} hours.")
                    break

        return progress_complete


def prepare_bizspeech_speechbrain(hparams: dict):
    """Entrypoint for speechbrain function that uses a method directly as a parameter.

    Args:
        hparams ([type]): [description]
    """
    data_folder = hparams["local_dataset_folder"]
    training_file = pathlib.Path(hparams["train_annotation"])
    valid_file = pathlib.Path(hparams["valid_annotation"])
    test_file = pathlib.Path(hparams["test_annotation"])
    if training_file.is_file() and valid_file.is_file() and test_file.is_file():
        logger.info(f"Training JSON/CSV files exists.")
        if hparams["use_wds"]:
            train_shards = [str(f) for f in sorted(
                pathlib.Path(data_folder).glob("train_shard-*.tar*"))]
            val_shards = [str(f) for f in sorted(
                pathlib.Path(data_folder).glob("val_shard-*.tar*"))]
            test_shards = [str(f) for f in sorted(
                pathlib.Path(data_folder).glob("test_shard-*.tar*"))]
            if train_shards and val_shards and test_shards:
                logger.info(
                    f"Using webdataset and tar shards are already available.")
    else:
        dest_dir = pathlib.Path(
            hparams["local_dataset_folder"]).resolve().parent
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
        datasetObj.parse_to_json(hparams["local_dataset_folder"])
        logger.info(
            f'Created JSON dataset files under {hparams["local_dataset_folder"]} directory.')
        if hparams["use_wds"]:
            logger.info("Starting to create tars compatible for webdataset")
            datasetObj.json_to_webdataset_tar(
                "train", hparams["local_dataset_folder"], hparams["number_of_shards"], hparams["use_compression"], hparams["preprocess_audio"])
            logger.info("Train tar shards written out!")
            datasetObj.json_to_webdataset_tar(
                "val", hparams["local_dataset_folder"], hparams["number_of_shards"], hparams["use_compression"], hparams["preprocess_audio"])
            logger.info("Validation tar shards written out!")
            datasetObj.json_to_webdataset_tar(
                "test", hparams["local_dataset_folder"], hparams["number_of_shards"], hparams["use_compression"], hparams["preprocess_audio"])
            logger.info("Test tar shards written out!")

    if hparams["copy_to_local"]:
        local_path = hparams["local_dest_dir"]
        local_train_shards = [str(f) for f in sorted(
                pathlib.Path(local_path).glob("train_shard-*.tar*"))]
        local_val_shards = [str(f) for f in sorted(
                pathlib.Path(local_path).glob("val_shard-*.tar*"))]
        local_test_shards = [str(f) for f in sorted(
                pathlib.Path(local_path).glob("test_shard-*.tar*"))]
        if local_train_shards and local_val_shards and local_test_shards:
            logger.info(f"Local drive already has the required data.")
        else:
            logger.info(f"Copying to local drive.")
            train_shards = [str(f) for f in sorted(
                    pathlib.Path(data_folder).glob("train_shard-*.tar*"))]
            val_shards = [str(f) for f in sorted(
                    pathlib.Path(data_folder).glob("val_shard-*.tar*"))]
            test_shards = [str(f) for f in sorted(
                    pathlib.Path(data_folder).glob("test_shard-*.tar*"))]
            dest_dir = pathlib.Path(local_path).resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            for shard_file in tqdm(train_shards + val_shards + test_shards):
                shutil.copy(shard_file, dest_dir)
    logger.info(f"Dataset is ready to use.")


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
