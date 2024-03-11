import warnings

import pandas as pd

from sklearn.model_selection import train_test_split

import pickle


def extract_pose_direction(input: list) -> list:
    if len(input) <= 0:
        return input

    output = []

    for row in input:
        pose_x = row[0]
        pose_y = row[1]

        # determine face direction
        if pose_y > 15:
            features = [3, pose_x, pose_y]  # right
        elif pose_y < -10:
            features = [2, pose_x, pose_y]  # left
        elif pose_x < -10:
            features = [1, pose_x, pose_y]  # down
        else:
            features = [0, pose_x, pose_y]  # forward

        output.append(features)

    return output


def compute_attentiveness(predictions: list) -> (float, int, int):
    positive = predictions.count(1)
    negative = predictions.count(0)

    return positive / (len(predictions)) * 100, positive, negative


def update_database(record: dict):
    pass


class AttentivenessClassifier:
    def __init__(self, model_path: str):
        try:
            self.model = pickle.load(open(model_path, "rb"))
        except (TypeError, OSError):
            self.model = None
            print("Failed to load model")

    def predict(self, x: list) -> list:
        return self.model.predict(x).tolist() if len(x) > 0 else None


if __name__ == "__main__":
    # todo replace with code to load data from mqtt or db
    data = pd.read_csv("data/attention_detection_dataset_processed.csv")
    X = data.drop(["label"], axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    classifier = AttentivenessClassifier("rf_model.sav")
    y_pred = classifier.predict(x_test)
    att_ratio, pos_count, neg_count = compute_attentiveness(y_pred)

    print(f"Number of 1s: {pos_count}")
    print(f"Number of 0s: {neg_count}")
    print(f"Audience Attentiveness: {round(att_ratio, 2)}%")
