Git Repository for using speechbrain on the Bizspeech data

# Data Prep

- [X]  Start with a small set

    What's "small"?

    Choose (20 hours * 2) for development and test sets.  == 80 Hours

    Pick from test set 1 and 2 (so we have approximate known WER) for these

- [X]  Choosing Data Distribution

    Effectively 4 sets of 20 hours each.

    Each one will have

    10 hours of native speakers and 10 hours of non native speaker

    5 hours of presentation and 5 hours of qna for each of these 10 hours

- [x]  Data Preprocessing (to use for training)

    Try to choose data without noise. Add exlusion list from Google Azure results which have very high WER (>100%)

    Split to reasonably short utterances (~ 1 sentence each). Important for attention models.

- [x]  Speechbrain Compatible Data Loading

    Newest method uses JSON

    Look at recipies from TIMIT and WS5 for reference on speechbrain repo github

# DL Phase

- [ ]  Attention based Encoder - Decoder model to start off

    CRDNN Model? CTC?

- [ ]  Hyperparameter settings from TIMIT recipies
- [ ]  Transformer models later?

    They are very sensitive to hyperparameters

    Maybe also use pretrained model?
