"""Recipe for training a sequence-to-sequence ASR system with mini-librispeech.

Authors
 * Anand Umashankar 2021

"""

import json
import logging
import pathlib
import string
import sys
import time
from logging.handlers import QueueHandler, QueueListener

import numpy as np
import sentencepiece as spm
import speechbrain as sb
import torch
import torch.multiprocessing as mp
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from bizspeech_prepare import prepare_bizspeech_speechbrain

logger = logging.getLogger(__name__)


def dataio_prepare_wds(hparams: dict):
    def data_pipeline(sample_dict: dict):
        txt = sample_dict["meta"]["txt"]
        tokens_list = hparams["tokenizer"].encode_as_ids(txt)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        tokens = torch.LongTensor(tokens_list)

        return {
            "id": sample_dict["id"],
            "sig": sample_dict["audio_tensor"],
            "words": txt,
            "tokens_list": tokens_list,
            "tokens_bos": tokens_bos,
            "tokens_eos": tokens_eos,
            "tokens": tokens
        }

    datasets = {}
    data_folder = hparams["local_dataset_folder"]
    shards_pattern = str(hparams["shardfiles_pattern"])
    for dataset in ["train", "val", "test"]:
        with open(hparams["local_dataset_folder"] + "/" + dataset + ".json") as f:
            length_of_set = len(json.load(f))
        if dataset == "train":
            train_shard_count = len([str(f) for f in sorted(
                pathlib.Path(data_folder).glob(dataset + "_shard-*.tar*"))])
            if hparams["use_dynamic_batch_size"]:
                datasets[dataset] = (
                    wds.WebDataset(
                        [shards_pattern % a for a in range(
                            hparams[dataset+"_shards"][0], hparams[dataset+"_shards"][1])],
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
                            hparams[dataset+"_shards"][0], hparams[dataset+"_shards"][1])],
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
            logger.info(f"{dataset}ing data consist of {length_of_set} samples")
        else:
            datasets[dataset] = (
                wds.WebDataset(
                    [shards_pattern % a for a in range(
                        hparams[dataset+"_shards"][0], hparams[dataset+"_shards"][1])],
                    length=688)
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

    return datasets, train_shard_count


def worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def update_average(step, loss, avg_loss):
    """Update running average of the loss.

    Arguments
    ---------
    loss : torch.tensor
        detached loss, a single float value.
    avg_loss : float
        current running average.

    Returns
    -------
    avg_loss : float
        The average loss.
    """
    if torch.isfinite(loss):
        avg_loss -= avg_loss / step
        avg_loss += float(loss) / step
    return avg_loss


def compute_objectives(predictions, batch, stage, hparams, epoch_counter, ctc_cost, tokenizer=None, wer_metric=None, cer_metric=None, wer_metric_p=None):
    """Compute the loss (CTC+NLL) given predictions and targets."""
    current_epoch = epoch_counter.current

    if stage == sb.Stage.TRAIN:
        if current_epoch <= hparams["number_of_ctc_epochs"]:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens = predictions
    else:
        p_seq, wav_lens, predicted_tokens = predictions

    ids = batch.id
    tokens_eos, tokens_eos_lens = batch.tokens_eos
    tokens, tokens_lens = batch.tokens

    loss_seq = sb.nnet.losses.nll_loss(
        log_probabilities=p_seq,
        targets=tokens_eos,
        length=tokens_eos_lens,
        label_smoothing=hparams["label_smoothing"],
    )

    # Add ctc loss if necessary
    if (
        stage == sb.Stage.TRAIN
        and current_epoch <= hparams["number_of_ctc_epochs"]
    ):
        loss_ctc = ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        )
        loss = hparams["ctc_weight"] * loss_ctc
        loss += (1 - hparams["ctc_weight"]) * loss_seq
    else:
        loss = loss_seq

    if stage != sb.Stage.TRAIN:
        table = str.maketrans('','',string.punctuation)
        predicted_words_p = [
            tokenizer.decode_ids(utt_seq).split(" ")
            for utt_seq in predicted_tokens
        ]
        target_words_p = [wrd.split(" ") for wrd in batch.words]
        predicted_words = [
                [a.upper().translate(table) for a in tokenizer.decode_ids(utt_seq).split(" ")]
                for utt_seq in predicted_tokens
            ]
        target_words = [[a.upper().translate(table) for a in wrd.split(" ")] for wrd in batch.words]
        wer_metric.append(ids, predicted_words, target_words)
        wer_metric_p.append(ids, predicted_words_p, target_words_p)
        cer_metric.append(ids, predicted_words, target_words)

    return loss


def compute_forward(batch, stage, device, hparams, modules, compute_features, epoch_counter, log_softmax, ctc_lin, valid_search, test_search):
    batch = batch.to(device)
    wavs, wav_lens = batch.sig

    tokens_bos, _ = batch.tokens_bos
    wavs, wav_lens = wavs.to(device), wav_lens.to(device)

    # Forward pass
    feats = compute_features(wavs)
    feats = modules.normalize(feats, wav_lens)
    x = modules.encoder(feats.detach())
    e_in = modules.embedding(tokens_bos)
    h, _ = modules.decoder(e_in, x, wav_lens)
    logits = modules.seq_lin(h)
    p_seq = log_softmax(logits)

    # Compute outputs
    if stage == sb.Stage.TRAIN:
        current_epoch = epoch_counter.current
        if current_epoch <= hparams["number_of_ctc_epochs"]:
            # Output layer for ctc log-probabilities
            ctc_logits = ctc_lin(x)
            p_ctc = log_softmax(ctc_logits)
            return p_ctc, p_seq, wav_lens
        else:
            return p_seq, wav_lens
    else:
        if stage == sb.Stage.VALID:
            p_tokens, scores = valid_search(x, wav_lens)
        else:
            p_tokens, scores = test_search(x, wav_lens)
        return p_seq, wav_lens, p_tokens


def fit(hparams, modules, epoch_counter, checkpointer, run_opts, sync_flags, rank, train_loss_sync, tb_logger, logger, train_set, valid_set):
    device = run_opts["device"]
    #modules = torch.nn.ModuleDict(hparams["modules"]).to(device)
    optimizer = hparams["opt_class"](modules.parameters())
    compute_features = hparams["compute_features"]
    log_softmax = hparams["log_softmax"]
    ctc_lin = hparams["ctc_lin"]
    valid_search = hparams["valid_search"]
    test_search = hparams["test_search"]
    tokenizer = hparams["tokenizer"]
    ctc_cost = hparams["ctc_cost"]
    avg_train_loss = 0.0

    # Iterate epochs
    for epoch in epoch_counter:
        logger.info("Entering epoch number " + str(epoch) + " in process " + str(rank))
        
        # Training stage
        modules.train()
        with tqdm(train_set, disable=rank != 0) as t:
            for step, batch in enumerate(t):
                predictions = compute_forward(batch, sb.Stage.TRAIN, device, hparams, modules,
                                            compute_features, epoch_counter, log_softmax, ctc_lin, valid_search, test_search)

                loss = compute_objectives(
                    predictions, batch, sb.Stage.TRAIN, hparams, epoch_counter, ctc_cost)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = loss.detach()
                avg_train_loss = update_average(step+1, loss, avg_train_loss)
                t.set_postfix(train_loss=avg_train_loss)
        
        train_loss_sync.append(avg_train_loss)
        sync_flags["validation_flag"] = True
        sync_flags["epoch_training_flag"] += 1

        avg_train_loss = 0.0
        if rank == 0:
            while not sync_flags["epoch_training_flag"] == hparams["num_processes"]:
                logger.info("Waiting for training epoch to complete in all processes.")
                time.sleep(10)
            
            logger.info("Starting validation in process 0")
            wer_metric = hparams["error_rate_computer"]()
            cer_metric = hparams["cer_computer"]()
            modules.eval()
            avg_valid_loss = 0.0
            step = 0
            with torch.no_grad():
                with tqdm(valid_set) as t:
                    for batch in t:
                        step += 1
                        predictions = compute_forward(batch, sb.Stage.VALID, device, hparams, modules,
                                                    compute_features, epoch_counter, log_softmax, ctc_lin, valid_search, test_search)
                        with torch.no_grad():
                            loss = compute_objectives(
                                predictions, batch, sb.Stage.VALID, hparams, epoch_counter, ctc_cost, tokenizer, wer_metric, cer_metric)

                        loss = loss.detach()
                        avg_valid_loss = update_average(step, loss, avg_valid_loss)
                        t.set_postfix(train_loss=avg_train_loss)
            step = 0
            stage_stats = {"loss": avg_valid_loss}
            stage_stats["WER"] = wer_metric.summarize("error_rate")
            stage_stats["CER"] = cer_metric.summarize("error_rate")


            train_stats = {"loss": np.mean(train_loss_sync)}
            hparams["train_logger"].log_stats(
                stats_meta={"epoch": epoch},
                train_stats=train_stats,
                valid_stats=stage_stats,
            )
            tb_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=train_stats,
                valid_stats=stage_stats,
            )
            checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
            train_loss_sync = []
            sync_flags["epoch_training_flag"] = 0
            sync_flags["validation_flag"] = False
        
        while sync_flags["validation_flag"]:
            time.sleep(10)

def eval(hparams, modules, epoch_counter, checkpointer, run_opts, sync_flags, rank, train_loss_sync, tb_logger, logger, test_set):
    device = run_opts["device"]
    #modules = torch.nn.ModuleDict(hparams["modules"]).to(device)
    compute_features = hparams["compute_features"]
    log_softmax = hparams["log_softmax"]
    ctc_lin = hparams["ctc_lin"]
    valid_search = hparams["valid_search"]
    test_search = hparams["test_search"]
    tokenizer = hparams["tokenizer"]
    ctc_cost = hparams["ctc_cost"]
    avg_train_loss = 0.0

    wer_metric = hparams["error_rate_computer"]()
    wer_metric_p = hparams["error_rate_computer"]()
    cer_metric = hparams["cer_computer"]()
    modules.eval()
    avg_valid_loss = 0.0
    step = 0
    with torch.no_grad():
        with tqdm(test_set) as t:
            for batch in t:
                step += 1
                predictions = compute_forward(batch, sb.Stage.VALID, device, hparams, modules,
                                            compute_features, epoch_counter, log_softmax, ctc_lin, valid_search, test_search)
                with torch.no_grad():
                    loss = compute_objectives(
                        predictions, batch, sb.Stage.VALID, hparams, epoch_counter, ctc_cost, tokenizer, wer_metric, cer_metric, wer_metric_p)

                loss = loss.detach()
                avg_valid_loss = update_average(step, loss, avg_valid_loss)
                t.set_postfix(train_loss=avg_train_loss)
    step = 0
    stage_stats = {"loss": avg_valid_loss}
    stage_stats["WER"] = wer_metric.summarize("error_rate")
    stage_stats["WER_p"] = wer_metric_p.summarize("error_rate")
    stage_stats["CER"] = cer_metric.summarize("error_rate")


    hparams["train_logger"].log_stats(
        stats_meta={"epoch": 1},
        test_stats=stage_stats,
    )

    checkpointer.save_and_keep_only(
        meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
    )
    


def train(rank, run_opts, hparams, modules, sync_flags, train_loss_sync, log_q):
    np.random.seed(np.random.get_state()[1][0] + rank)
    torch.manual_seed(int(hparams["seed"]) + rank)

    qh = QueueHandler(log_q)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(qh)

    epoch_counter = hparams["epoch_counter"]
    checkpointer = hparams["checkpointer"]
    datasets, train_shard_count = dataio_prepare_wds(hparams)
    tokenizer = spm.SentencePieceProcessor(
        model_file="runs/98585/save/5000_bpe.model")
    hparams["tokenizer"] = tokenizer

    if hparams["use_dynamic_batch_size"]:
        hparams["train_dataloader_opts"]["looped_nominal_epoch"] = hparams["looped_nominal_epoch"]
        datasets["train"] = sb.dataio.dataloader.make_dataloader(
            datasets["train"], **hparams["train_dataloader_opts"])

    if rank == 0:
        tb_logger = sb.utils.train_logger.TensorboardLogger(save_dir=hparams["tensorboard_dir"])
    else:
        tb_logger = None

    fit(
        hparams,
        modules,
        epoch_counter,
        checkpointer,
        run_opts,
        sync_flags,
        rank,
        train_loss_sync,
        tb_logger,
        logger,
        datasets["train"],
        datasets["val"]
    )

    eval(    
        hparams,
        modules,
        epoch_counter,
        checkpointer,
        run_opts,
        sync_flags,
        rank,
        train_loss_sync,
        tb_logger,
        logger,
        datasets["test"]
        )

def logger_init():
    q = mp.Queue()
    # this is the handler for all log records
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, handler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)

    return ql, q


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    q_listener, log_q = logger_init()
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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

    hparams.pop('dataset', None)

    manager = mp.Manager()
    sync_flags = manager.dict({"validation_flag":False, "epoch_training_flag":0})
    train_loss_sync = manager.list([])
    
    hparams["checkpointer"].recover_if_possible(device=torch.device(run_opts["device"]))
    modules = torch.nn.ModuleDict(hparams["modules"]).to(run_opts["device"])
    modules.share_memory()
    #for sub_model in hparams["model"]:
    #    sub_model.to(run_opts["device"])
    #    sub_model.share_memory()
    
    processes = []
    for rank in range(hparams["num_processes"]):
        p = mp.Process(target=train, args=(rank, run_opts, hparams, modules, sync_flags, train_loss_sync, log_q))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

