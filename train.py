"""Recipe for training a sequence-to-sequence ASR system with mini-librispeech.

Authors
 * Anand Umashankar 2021

"""

import json
import logging
import os
import pathlib
import shutil
import string
import sys
import traceback

import numpy as np
import speechbrain as sb
import torch
import torchaudio
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml

from data_prepare.bizspeech_prepare import prepare_bizspeech_speechbrain
from data_prepare.spgispeech import dataio_prepare_spgi
from data_prepare.librispeech import dataio_prepare_libri

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Run all the computation of the CTC + seq2seq ASR.

        It returns the posterior probabilities of the CTC and seq2seq networks.

        Arguments:
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns:
        -------
        predictions : dict
            At training time it returns predicted seq2seq log probabilities.
            If needed it also returns the ctc output log probabilities.
            At validation/test time, it returns the predicted tokens as well.

        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        # torchaudio.save(str(i)+"test.wav", wavs.to("cpu"), sample_rate=hparams["sample_rate"], channels_first=True)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.encoder(feats.detach())
        e_in = self.modules.embedding(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.decoder(e_in, x, wav_lens)
        # make_dot(self.modules.encoder(feats.detach()), params=dict(self.modules.encoder.named_parameters())).render("enc", format="png")
        # make_dot(self.modules.decoder(e_in, x, wav_lens), params=dict(self.modules.decoder.named_parameters())).render("dec", format="png")
        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)
        if self.hparams.train_WER_required:
            p_tokens, scores = self.hparams.valid_search(x, wav_lens)
        # Compute outputs
        if stage == sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                # Output layer for ctc log-probabilities
                ctc_logits = self.modules.ctc_lin(x)
                p_ctc = self.hparams.log_softmax(ctc_logits)
                if self.hparams.train_WER_required:
                    return p_ctc, p_seq, wav_lens, p_tokens
                else:
                    return p_ctc, p_seq, wav_lens
            else:
                if self.hparams.train_WER_required:
                    return p_seq, wav_lens, p_tokens
                else:
                    return p_seq, wav_lens
        else:
            if stage == sb.Stage.VALID and not self.hparams.train_WER_required:
                p_tokens, scores = self.hparams.valid_search(x, wav_lens)
            else:
                p_tokens, scores = self.hparams.test_search(x, wav_lens)
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Compute the loss (CTC+NLL) given predictions and targets."""
        current_epoch = self.hparams.epoch_counter.current

        if stage == sb.Stage.TRAIN:
            if current_epoch <= self.hparams.number_of_ctc_epochs:
                if self.hparams.train_WER_required:
                    p_ctc, p_seq, wav_lens, predicted_tokens = predictions
                else:
                    p_ctc, p_seq, wav_lens = predictions
            else:
                if self.hparams.train_WER_required:
                    p_seq, wav_lens, predicted_tokens = predictions
                else:
                    p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = sb.nnet.losses.nll_loss(
            log_probabilities=p_seq,
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )
        # self.hparams.seq_cost(
        #    p_seq, tokens_eos, length=tokens_eos_lens
        # )

        # Add ctc loss if necessary
        if (
            stage == sb.Stage.TRAIN
            and current_epoch <= self.hparams.number_of_ctc_epochs
        ):
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            )
            loss = self.hparams.ctc_weight * loss_ctc
            loss += (1 - self.hparams.ctc_weight) * loss_seq
        else:
            loss = loss_seq

        if stage == sb.Stage.TRAIN and self.hparams.train_WER_required:
            predicted_words = [
                self.hparams.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.words]
            self.wer_metric.append(ids, predicted_words, target_words)
        if stage != sb.Stage.TRAIN:
            table = str.maketrans('', '', string.punctuation)
            predicted_words_p = [
                self.hparams.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            predicted_words = [
                [a.upper().translate(table)
                 for a in self.hparams.tokenizer.decode_ids(utt_seq).split(" ")]
                for utt_seq in predicted_tokens
            ]
            target_words_p = [wrd.split(" ") for wrd in batch.words]
            target_words = [[a.upper().translate(table)
                             for a in wrd.split(" ")] for wrd in batch.words]
            self.wer_metric_p.append(ids, predicted_words_p, target_words_p)
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            if stage == sb.Stage.TEST and self.hparams.require_native_wer:
                if "non" in batch.category[0]:
                    self.wer_metric_nonnative.append(
                        ids, predicted_words, target_words)
                else:
                    self.wer_metric_native.append(
                        ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input."""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        # if self.hparams.gradient_accumulation:
        #    loss /= self.hparams.subbatches_count_for_grad_acc
        loss.backward()
        if not self.hparams.gradient_accumulation or (self.hparams.gradient_accumulation and self.step % self.hparams.subbatches_count_for_grad_acc == 0):
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Compute needed for validation/test batches."""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Run at the beginning of each epoch."""
        if stage == sb.Stage.TRAIN and self.hparams.train_WER_required:
            self.wer_metric = self.hparams.error_rate_computer()
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()
            self.wer_metric_p = self.hparams.error_rate_computer()
            self.cer_metric = self.hparams.cer_computer()
        if stage == sb.Stage.TEST and self.hparams.require_native_wer:
            self.wer_metric_native = self.hparams.error_rate_computer()
            self.wer_metric_nonnative = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Run at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            if self.hparams.train_WER_required:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["WER_p"] = self.wer_metric_p.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            if stage == sb.Stage.TEST and self.hparams.require_native_wer:
                stage_stats["WER_native"] = self.wer_metric_native.summarize(
                    "error_rate")
                stage_stats["WER_nonnative"] = self.wer_metric_nonnative.summarize(
                    "error_rate")
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if self.hparams.log_to_tensorboard:
                self.hparams.tensorboard_logger.log_stats(
                    stats_meta={"epoch": epoch, "lr": old_lr},
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if self.hparams.log_to_tensorboard:
                self.hparams.tensorboard_logger.log_stats(
                    stats_meta={
                        "Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=stage_stats,
                )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
            with open(self.hparams.wer_file_p, "w") as w:
                self.wer_metric_p.write_stats(w)
            if self.hparams.require_native_wer:
                with open(self.hparams.wer_native_file, "w") as w:
                    self.wer_metric_native.write_stats(w)
                with open(self.hparams.wer_nonnative_file, "w") as w:
                    self.wer_metric_nonnative.write_stats(w)


def dataio_prepare(hparams):
    """Prepare the datasets to be used in the brain class.

    It also defines the data processing pipeline through user-defined functions.

    Arguments:
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns:
    -------
    datasets : dict
        Dictionary containing "train", "val", and "test" keys that correspond
        to the DynamicItemDataset objects.

    """
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

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("audio", "start", "end")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(audio, start, end):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        if hparams["copy_to_local"]:
            event_id = str(pathlib.Path(audio).resolve().parent).split("/")[-1]
            dest_dir = pathlib.Path(
                hparams["local_dest_dir"]).resolve().joinpath(event_id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_dir = dest_dir.joinpath(
                "recording." + hparams["dataset"]["audio_filetype"])
            if not dest_dir.is_file():
                audio = shutil.copy(audio, dest_dir)

        # resample_reqd = False
        # mono_conversion = False
        if isinstance(start, str):
            metadata = torchaudio.info(audio)
        #    if metadata.sample_rate != hparams["sample_rate"]:
        #        resample_reqd = True
        #        effects.append(['rate', str(hparams["sample_rate"])])
        #    if metadata.num_channels != hparams["num_channels"]:
        #        mono_conversion = True
        #        effects.append(['channels', str(hparams["num_channels"])])
            start = int(time_str_to_seconds(start) * metadata.sample_rate)
            end = int(time_str_to_seconds(end) * metadata.sample_rate)
        num_frames = end - start

        audio, fs = torchaudio.load(
            audio, num_frames=num_frames, frame_offset=start)
        # a = time.time()
        audio = audio.transpose(0, 1)
        sig = hparams["preprocess_audio"](audio, metadata.sample_rate)
        # print("preprocess", time.time()-a)
        # if len(effects) > 0:
        #   audio, fs = torchaudio.sox_effects.apply_effects_tensor(audio, fs, effects)
        # if resample_reqd:
        #    audio = torchaudio.transforms.Resample(orig_freq=metadata.sample_rate, new_freq=hparams["sample_rate"]).forward(audio)
        # if mono_conversion:
        #    audio = torch.mean(audio, dim=0).unsqueeze(0)
        # audio = audio.transpose(0, 1)
        # sig = audio.squeeze(1)
        return sig
        # sig = sb.dataio.dataio.read_audio(
        #    {"file": audio, "start": start, "stop": end})
        # return sig

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with BOS are used for feeding
    # decoder during training, the tokens with EOS for computing the cost function.
    # The tokens without BOS or EOS is for computing CTC loss.
    @sb.utils.data_pipeline.takes("txt")
    @sb.utils.data_pipeline.provides(
        "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(txt):
        """Process the transcriptions to generate proper labels."""
        yield txt
        tokens_list = hparams["tokenizer"].encode_as_ids(txt)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # Define datasets from json data manifest file
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_folder = hparams["dataset"]["data_folder"]
    for dataset in ["train", "val", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams["dataset"][f"{dataset}_annotation"],
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=[
                "id",
                "sig",
                "words",
                "tokens_bos",
                "tokens_eos",
                "tokens",
            ],
        )
        hparams[f"{dataset}_dataloader_opts"]["shuffle"] = False

    # Sorting traiing data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["dataset"]["sorting"] == "ascending":
        # datasets["train"] = datasets["train"].filtered_sorted(
        #    sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["dataset"]["sorting"] == "descending":
        # datasets["train"] = datasets["train"].filtered_sorted(
        #    sort_key="length", reverse=True)
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["dataset"]["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    return datasets


def dataio_prepare_wds(hparams: dict):
    def data_pipeline(sample_dict: dict):
        txt = sample_dict["meta"]["txt"]
        tokens_list = hparams["tokenizer"].encode_as_ids(txt)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        tokens = torch.LongTensor(tokens_list)

        return {
            "id": sample_dict["id"],
            "category": sample_dict["meta"]["category"],
            "sig": sample_dict["audio_tensor"],
            "words": txt,
            "tokens_list": tokens_list,
            "tokens_bos": tokens_bos,
            "tokens_eos": tokens_eos,
            "tokens": tokens
        }

    datasets = {}
    if hparams["dataset"]["copy_to_local"]:
        data_folder = hparams["dataset"]["local_dest_dir"]
    else:
        data_folder = hparams["dataset"]["local_dataset_folder"]
    shards_pattern = str(hparams["dataset"]["shardfiles_pattern"])
    for dataset in ["train", "val", "test"]:
        with open(hparams["dataset"]["local_dataset_folder"] + "/" + dataset + ".json") as f:
            length_of_set = len(json.load(f))
        if dataset == "train":
            train_shard_count = len([str(f) for f in sorted(
                pathlib.Path(data_folder).glob(dataset + "_shard-*.tar*"))])
            if hparams["use_dynamic_batch_size"]:
                datasets[dataset] = (
                    wds.WebDataset(
                        [shards_pattern % a for a in range(
                            hparams["dataset"][dataset+"_shards"][0], hparams["dataset"][dataset+"_shards"][1])],
                        length=length_of_set)
                    .decode()
                    .rename(id="__key__", audio_tensor="wav.pyd", meta="meta.json")
                    .map(data_pipeline)
                    .then(
                        sb.dataio.iterators.dynamic_bucketed_batch,
                        **hparams["dynamic_batch_kwargs"],
                        # wds.iterators.batched,
                        # batchsize=hparams["batch_size"],
                        # collation_fn=sb.dataio.batch.PaddedBatch,
                        # partial=True
                    )
                )
            else:
                datasets[dataset] = (
                    wds.WebDataset(
                        [shards_pattern % a for a in range(
                            hparams["dataset"][dataset+"_shards"][0], hparams["dataset"][dataset+"_shards"][1])],
                        length=length_of_set)
                    .decode()
                    .rename(id="__key__", audio_tensor="wav.pyd", meta="meta.json")
                    .map(data_pipeline)
                    .then(
                        wds.iterators.batched,
                        batchsize=hparams["batch_size"],
                        collation_fn=sb.dataio.batch.PaddedBatch,
                        partial=True
                    )
                )
        else:
            datasets[dataset] = (
                wds.WebDataset(
                    [shards_pattern % a for a in range(
                        hparams["dataset"][dataset+"_shards"][0], hparams["dataset"][dataset+"_shards"][1])],
                    length=174)
                .decode()
                .rename(id="__key__", audio_tensor="wav.pyd", meta="meta.json")
                .map(data_pipeline)
                .then(
                    wds.iterators.batched,
                    batchsize=hparams["batch_size"],
                    collation_fn=sb.dataio.batch.PaddedBatch,
                    partial=True
                )
            )
        logger.info(f"{dataset}ing data consist of {length_of_set} samples")

    return datasets, train_shard_count


def worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if hparams["other_datasets"]["test_on_otherdatasets"] == "libri":
        test_datasets = dataio_prepare_libri(
            hparams["other_datasets"]["libri"])
    elif hparams["other_datasets"]["test_on_otherdatasets"] == "spgi":
        test_datasets = dataio_prepare_spgi(
            hparams["other_datasets"]["spgi"])
    else:
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
                "hparams": hparams["dataset"]
            })

        # We can now directly create the datasets for training, val, and test
        if hparams["dataset"]["use_wds"]:
            datasets, train_shard_count = dataio_prepare_wds(hparams)
        else:
            datasets = dataio_prepare(hparams)

    sb.utils.distributed.run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    if hparams["other_datasets"]["test_on_otherdatasets"] == "none":
        hparams["train_dataloader_opts"]["worker_init_fn"] = worker_init
        if hparams["use_dynamic_batch_size"]:
            hparams["train_dataloader_opts"]["looped_nominal_epoch"] = hparams["looped_nominal_epoch"]
            datasets["train"] = asr_brain.make_dataloader(
                datasets["train"], sb.Stage.TRAIN, **hparams["train_dataloader_opts"]
            )
            asr_brain.train_loader = datasets["train"]

        try:
            asr_brain.fit(
                asr_brain.hparams.epoch_counter,
                datasets["train"],
                datasets["val"],
                train_loader_kwargs=hparams["train_dataloader_opts"],
                valid_loader_kwargs=hparams["valid_dataloader_opts"],
            )
        except Exception as e:
            with open("debugging_info" + run_opts["device"] + ".txt", "a") as f:
                f.write(str(e))
                traceback.print_exc(file=f)

        # Load best checkpoint (highest STOI) for evaluation
        test_stats = asr_brain.evaluate(
            test_set=datasets["test"],
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
    else:
        print("here")
        for k in test_datasets.keys():  # keys are test_clean, test_other etc
            print(k, test_datasets[k])
            asr_brain.hparams.wer_file = os.path.join(
                hparams["dataset"]["output_folder"], "wer_{}.txt".format(k)
            )
            asr_brain.evaluate(
                test_datasets[k], test_loader_kwargs=hparams["other_datasets"]["test_dataloader_opts"]
            )
