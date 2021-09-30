import csv
import torch
import logging
import speechbrain as sb
from pathlib import Path

logger = logging.getLogger(__name__)

def dataio_prepare_spgi(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]
    
    # test is separate
    test_datasets = {}
    csv_filename = hparams["test_csv"]
    with open(csv_filename) as csv_file:
        result = {}
        reader = csv.DictReader(csv_file, delimiter="|", skipinitialspace=True)
        for row in reader:            
            data_id = row["wav_filename"]
            row["wav_filename"] = data_folder + row["wav_filename"]
            result[data_id] = row
        spgi_data = sb.dataio.dataset.DynamicItemDataset(result)
        test_datasets = {"test_spgi" : spgi_data}

    #datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    datasets = [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_filename")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_filename):
        sig = sb.dataio.dataio.read_audio(wav_filename)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides(
        "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(words):
        yield words
        tokens_list = tokenizer.encode_as_ids(words)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "words", "tokens_bos", "tokens_eos", "tokens"],
    )
    return test_datasets