def calculateMetrics(output, groundtruth, label):
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    for i, (prediction_output, actual_output) in enumerate(zip(output, groundtruth)):
        if prediction_output[label] == 1 and actual_output[label] == 1:
            true_positives += 1
        elif prediction_output[label] == 0 and actual_output[label] == 0:
            true_negatives += 1
        elif prediction_output[label] == 1 and actual_output[label] == 0:
            false_positives += 1
        else:
            false_negatives += 1

    return true_positives, true_negatives, false_positives, false_negatives


def displayMetrics(tp, tn, fp, fn):
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    if tp + fp != 0.0:
        precision = tp / (tp + fp)
    if tp + fn != 0.0:
        recall = tp / (tp + fn)
    if (precision + recall) != 0.0:
        f1 = (2 * precision * recall) / (precision + recall)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)


def evaluateModel(output, groundtruth):
    # for label seek
    # Report,Device,Delivery,Progress,becoming_member,attempt_action,Activity,Other
    tp, tn, fp, fn = calculateMetrics(output, groundtruth, 0)
    print("Metrics for Report Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, 1)
    print("Metrics for Device Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, 2)
    print("Metrics for Delivery Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, 3)
    print("Metrics for Progress Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, 4)
    print("Metrics for becoming_member Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, 5)
    print("Metrics for attempt_action Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, 6)
    print("Metrics for Activity Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, 7)
    print("Metrics for Other Frame")
    displayMetrics(tp, tn, fp, fn)


def main():
    output = [[1], [0], [1], [0]]
    ground_truth = [[1], [1], [0], [0]]

    tp, tn, fp, fn = calculateMetrics(output, ground_truth, 0)
    print("Metrics for Delivery Frame")
    displayMetrics(tp, tn, fp, fn)

    pass


if __name__ == '__main__':
    main()