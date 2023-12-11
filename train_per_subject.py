import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

cube_size = 5

ROIs = [{"name": "left_amygdala", "coord": (-30, -4, -22)}, 
        {"name": "right_amygdala", "coord": (30, -4, -22)}, 
        {"name": "right_insula", "coord": (42, -2, 4)},
        {"name": "left_insula", "coord": (-42, -2, 4)},
        {"name": "periaqueductal", "coord": (0, -30, -10)},
        {"name": "left_ventral_striatum", "coord": (-10, 10, -6)},
        {"name": "right_ventral_striatum", "coord": (10, 10, -6)},
        {"name": "left_putamen", "coord": (-24, 0, 4)},
        {"name": "right_putamen", "coord": (24, 0, 4)},
        {"name": "anterior_cingulate", "coord": (0, 30, 18)},
        {"name": "ventromedial_prefronal", "coord": (0, 42, -12)},
        {"name": "ventral_tegmental", "coord": (4, -18, -14)},
        {"name": "V6", "coord": (9, -82, 36)},
        {"name": "V1", "coord": (-4, -88, -2)}, 
        {"name": "noise", "coord": (-60, -90, -44)}] # bottom left corner

def load_labels(width, lang):

    sentiment_df = pd.read_csv("/home/moemen/CMPUT624/labels/new_"+str(lang)+"_labels_with_timing.csv")

    # Extract only the probability columns for processing
    prob_cols = ['positive', 'negative', 'neutral']

    THRESHOLD = 0.6

    # Function to convert probabilities to 1 for the maximum probability and 0 for others
    def convert_to_binary(row):
        if max(row[prob_cols]) < THRESHOLD:
            row[prob_cols] = False
            return row
        max_col = row[prob_cols].idxmax()
        row[prob_cols] = False
        row[max_col] = True
        return row

    def should_be_kept(row):
        if max(row[prob_cols]) == 0:
            return False
        return True
    
    def voting_filter(df, label_columns, window_size):
        result = pd.DataFrame(index=df.index, columns=label_columns)

        for i in range(0, len(df), window_size):
            window = df.iloc[i:i+window_size]

            if not window.empty:
                # Count occurrences of each label in the window
                label_counts = window[label_columns].sum()

                # Find the most common label
                if max(label_counts) == 0:
                    result.loc[window.index] = 0
                    continue
                most_common_label = label_counts.idxmax()

                # Assign the most common label to all rows in the window
                result.loc[window.index] = 0
                result.loc[window.index, most_common_label] = 1

        return result
    
    # Apply the function to each row
    sentiment_df = sentiment_df.apply(convert_to_binary, axis=1)
    if width != 0:
        sentiment_df[prob_cols] = voting_filter(sentiment_df, prob_cols, width)
    sentiment_df = sentiment_df[sentiment_df.apply(should_be_kept, axis=1)]

    num_pos = len(sentiment_df[sentiment_df["positive"] == 1])
    num_neg = len(sentiment_df[sentiment_df["negative"] == 1])
    num_neut = len(sentiment_df[sentiment_df["neutral"] == 1])

    remove_n = num_neut - num_neg
    drop_indices = np.random.choice(sentiment_df[sentiment_df['neutral'] == 1].index, remove_n, replace=False)
    sentiment_df = sentiment_df.drop(drop_indices)

    sentiment_df["t1"] = sentiment_df["t1"].apply(lambda x: x/2).apply(np.ceil)
    sentiment_df["t2"] = sentiment_df["t2"].apply(lambda x: x/2).apply(np.ceil)

    return sentiment_df

def load_and_match_roi_data(roi_folder, sentiment_df, num_TRs, start_from):
    fmri_files = [os.path.join(roi_folder, file) for file in os.listdir(roi_folder)]
    fmri_data = [np.load(file) for file in fmri_files]
    X = np.empty(shape=(0,cube_size*cube_size*cube_size*num_TRs))
    y = np.empty(shape=(0, 3))
    for row in sentiment_df.iterrows():
        t2 = int(row[1]['t2'])
        section = int(row[1]['section'])
        fmri_file = fmri_data[section]
        fmri_duration = fmri_file.shape[-1]
        relevant_fmri_data = []
        not_enough = False
        for i in range(start_from, num_TRs+start_from):
            if t2+i >= fmri_duration:
                not_enough = True
                continue
            relevant_fmri_data.append(fmri_file[:,:,:, t2+i])
        if not_enough:
            continue
        relevant_fmri_data = np.array(relevant_fmri_data)
        labels = np.array([row[1]['positive'], row[1]['negative'], row[1]['neutral']])
    #    mean_fmri = np.mean(relevant_fmri_data, axis=0).reshape(cube_size*cube_size*cube_size,)
        concat_fmri = relevant_fmri_data.reshape(cube_size*cube_size*cube_size*num_TRs,)
        X = np.vstack((X, concat_fmri))
        y = np.vstack((y, labels))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def train_and_evaluate_ROI(roi, processed_folder, train_lang, num_TRs, start_from, width):
    train_data_folders = [os.path.join(processed_folder, folder_name) for folder_name in os.listdir(processed_folder) if train_lang in folder_name]
    print("Training on ROI:", roi["name"])

    accuracies = []
    for i in range(len(train_data_folders)):
        
        train_subject_folder = train_data_folders[i]
        print("Subject", i+1, "/", len(train_data_folders))

        # Load and preprocess sentiment_df
        sentiment_df = load_labels(width, train_lang)

        # Load roi data from train subject
        X = np.empty(shape=(0,cube_size*cube_size*cube_size*num_TRs))
        y = np.empty(shape=(0, 3))

        roi_folder = os.path.join(train_subject_folder, "ROIs", roi["name"])
        X, y = load_and_match_roi_data(roi_folder, sentiment_df, num_TRs, start_from)

        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


        # Balance and preprocess the data
        y_train = np.argmax(y_train.astype(int), axis=1)
        y_test = np.argmax(y_test.astype(int), axis=1)
        ros_train = RandomOverSampler(random_state=42)
        ros_test = RandomOverSampler(random_state=42)
        X_train, y_train = ros_train.fit_resample(X_train, y_train)
        X_test, y_test = ros_test.fit_resample(X_test, y_test)

        # Define and train model
        classifier = OneVsOneClassifier(LogisticRegression(random_state=42, penalty='l2', max_iter=2000))
        classifier.fit(X_train, y_train)
        print(classifier.score(X_train, y_train))

        # Evaluate model on test data
        y_pred = classifier.predict(X_test)
        score = f1_score(y_test, y_pred, average='micro')
        accuracies.append(score)

    # Return accuracies
    return accuracies


if __name__ == "__main__":
    train_lang = sys.argv[1]
    processed_folder = "/media/moemen/Stuff/project/data/processed/smoothing_0"
    results_folder = "/media/moemen/Stuff/project/data/results"
    results = {}
    filter_width = [0, 3, 5, 7]
    num_TRs_list = [1, 2, 3]
    start_from_list = [-1, 0, 1]

    for width in filter_width:
        if width == 0:
            results_file = os.path.join(results_folder, 'smoothing_0', 'per_subject_unshuffled', "results_"+train_lang+"_concat_no_filter.json")
        else:
            results_file = os.path.join(results_folder, 'smoothing_0', 'per_subject_unshuffled', "results_"+train_lang+"_concat_filter"+str(width)+".json")

        for roi in ROIs:
            results[roi["name"]] = {}
            for num_TRs in num_TRs_list:
                results[roi["name"]]["num_TRs_"+str(num_TRs)] = {}
                for start_from in start_from_list:
                    accuracies = train_and_evaluate_ROI(roi, processed_folder, train_lang, num_TRs, start_from, width)
                    results[roi["name"]]["num_TRs_"+str(num_TRs)]["start_from_"+str(start_from)] = accuracies
    
        with open(results_file, 'w') as out_file:
            json.dump(results, out_file)



    

    
