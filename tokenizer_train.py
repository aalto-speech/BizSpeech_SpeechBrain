"""Script for training a BPE tokenizer on the top of CSV or JSON annotation files.

The tokenizer converts words into sub-word units that can be used to train a
language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bad results when combining AM and LM.
"""
import sys
import pathlib
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from bizspeech_prepare import prepare_bizspeech_speechbrain


logger = logging.getLogger(__name__)


def train_tokenizer(hparams):
    """Train tokenizer. Checks if a model already exists."""
    hparams["tokenizer_train"]()
    logger.info("Tokenizer trained!")


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    print(hparams)
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["dataset"]["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_bizspeech_speechbrain,
        kwargs={
            "local_dataset_folder": hparams["dataset"]["local_dataset_folder"],
            "data_folder": hparams["dataset"]["data_folder"],
            "hours_reqd": hparams["dataset"]["hours_reqd"],
            "nonnative": hparams["dataset"]["nonnative"],
            "qna": hparams["dataset"]["qna"],
            "strict_included": hparams["dataset"]["strict_included"],
            "non_CEO_utt": hparams["dataset"]["non_CEO_utt"],
            "seed": hparams["dataset"]["seed"],
            "trainValTest": hparams["dataset"]["trainValTest"],
            "output_format": hparams["dataset"]["output_format"],
            "include_event_json": hparams["dataset"]["include_event_json"],
            "exclude_event_json": hparams["dataset"]["exclude_event_json"],
            "utterance_duration_limit": hparams["dataset"]["utterance_duration_limit"],
            "audio_filetype": hparams["dataset"]["audio_filetype"]
        },
    )

    train_tokenizer(hparams)
