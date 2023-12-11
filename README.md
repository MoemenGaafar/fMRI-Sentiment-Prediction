# Comparing Emotional Processing in Speakers of Different Languages
According to the Conceptual Act Theory, our emotions are not only affected by external stimuli and our bodies' physical responses to them, but are also affected by the way we label and categorize these experiences. Based on this, we hypothesize that language has an effect on how our brains process emotions. In this project, we investigate this hypothesis by studying the fMRI scans of subjects who speak two different languages. The subjects listen to the same audiobook but in their own native language. To study this correlation, we train decoding models to predict the sentiment label of a sentence given the fMRI voxel activity corresponding to it. Even though our results are not conclusive, we find that there are more correlations among subjects that speak the same language compared to subjects that speak different languages.

You can download and inspect our results directly [here](https://github.com/MoemenGaafar/fMRI-Sentiment-Prediction/blob/main/results.zip).

# To replicate our results
1) Download [Le Petit Prince Dataset](https://openneuro.org/datasets/ds003643/versions/2.0.0).
2) [Optional] Inspect the data by running [inspect_data.ipynb](https://github.com/MoemenGaafar/fMRI-Sentiment-Prediction/blob/main/inspect_data.ipynb).
3) Extract the sentiment labels from sentences by running [labels.ipynb](https://github.com/MoemenGaafar/fMRI-Sentiment-Prediction/blob/main/labels.ipynb).
4) Match timing data with the sentiment labels by running [add_timing.ipynb](https://github.com/MoemenGaafar/fMRI-Sentiment-Prediction/blob/main/add_timing.ipynb).
5) [Optional] Run [model_train.ipynb](https://github.com/MoemenGaafar/fMRI-Sentiment-Prediction/blob/main/model_train.ipynb) to inspect the process of preprocessing the data and training the models.
6) Run [preprocess.py](https://github.com/MoemenGaafar/fMRI-Sentiment-Prediction/blob/main/preprocess.py) with either argument "EN" or "FR" to preprocess and save ROI data to disk.
7) Run [train.py](https://github.com/MoemenGaafar/fMRI-Sentiment-Prediction/blob/main/train.py) which takes two arguments corresponding to the train and test languages. This replicates the LOSO study results.
8) Run [train_per_subject.py](https://github.com/MoemenGaafar/fMRI-Sentiment-Prediction/blob/main/train_per_subject.py) which takes one arguments corresponding to the train and test language. This replicates the per-subject study results.

# General Notes
1) All files have references to file paths. You need to change these references to where you saved your data.
2) preprocess.py allows you to change the smoothing through variables in the __main__ method.
3) train.py and train_per_subject.py allow you to change the aggregation function and smoothing through variables in the __main__ method.


