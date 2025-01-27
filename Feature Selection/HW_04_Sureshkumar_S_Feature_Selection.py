"""
Author - Suraj Sureshkumar(ss7495)
Homework 4
Description - A program for best feature selection
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def processing(file):
    """
    Reading the file and creating a dataframe
    :param filename: file for the dataframe to be created
    :return: dataframe
    """
    dataframe = pd.read_csv(file)
    return dataframe


def computation(dataframe):
    """
    Computing the min, max, avg, sd and median growth
    :param dataframe: original dataframe
    :return: computed features dataframe
    """
    # Dropping the cls column
    data_without_cls = dataframe.drop(columns='Cls')
    # finding the difference for each column which is column2 - column1 and so on
    # here .diff does this for us
    growth_per_day = data_without_cls.diff(axis=1).drop(columns=data_without_cls.columns[0])

    # lines 35-39 we are using the inbuilt min,max,mean, std, median functions to compute the values for point 5
    minimum_growth = growth_per_day.min(axis=1)
    maximum_growth = growth_per_day.max(axis=1)
    average_growth = growth_per_day.mean(axis=1)
    standard_deviation = growth_per_day.std(axis=1)
    median_growth = growth_per_day.median(axis=1)

    # creating a new data frame with the computed values and the adding the class back to the dataframe
    computed_features = pd.DataFrame({
        "Cls": dataframe['Cls'],
        "Minimum Growth": minimum_growth,
        'Maximum Growth': maximum_growth,
        "Average Growth": average_growth,
        "Standard Deviation": standard_deviation,
        "Median Growth": median_growth
    })

    return computed_features


def feature_selection_and_threshold(dataframe):
    """
    Selecting the best feature and the threshold
    :param dataframe: computed dataframe
    :return: best_feature and best_threshold
    """
    minimum_mistakes = float('inf')  # initial value
    best_feature = 0
    best_threshold = 0

    # looping through each feature in the dataframe
    for features in dataframe:
        # if the encountered feature is Cls then we continue ahead
        if features == 'Cls':
            continue
        # if not then get the min value and max value of that column of that particular feature
        min_value = dataframe[features].min()
        max_value = dataframe[features].max()

        # finding the delta value
        delta_value = (max_value - min_value) / 20

        # start and stop value for the for loop
        start = min_value - delta_value
        stop = max_value - delta_value

        # looping through the start and stop with a step of the delta_value
        for threshold in np.arange(start, stop, delta_value):
            total_mistakes = 0  # variable to store the mistakes
            #  creating a column in data which holds the value 0 or 1 if the current features value is greater than the
            # threshold
            dataframe['Predicted Values'] = np.where(dataframe[features] > threshold, 0, 1)
            # looping through the two columns and checking each value
            for value1, value2 in zip(dataframe['Cls'], dataframe['Predicted Values']):
                # if the value 1 and value 2 are not same then increment mistakes
                if value1 == 1 and value2 != 1:
                    total_mistakes += 1
                # if the value 1 is 2 and value 2 is not 0 then increment mistakes
                elif value1 == 2 and value2 != 0:
                    total_mistakes += 1
                # if the value 1 is 3 and value 2 is not 0 then increment mistakes
                elif value1 == 3 and value2 != 0:
                    total_mistakes += 1
                else:
                    # if both are same then continue
                    continue

            # checking if mistakes is less than minimum mistakes then set the current mistakes as the minimum mistakes
            # and the best feature is the feature and the best threshold to threshold
            if total_mistakes <= minimum_mistakes:
                minimum_mistakes = total_mistakes
                best_feature = features
                best_threshold = threshold

    return best_feature, best_threshold


def plot(dataframe, best_threshold):
    """
    Plotting the histogram
    :param dataframe: data frame
    :param best_threshold: the best threshold
    :return: none
    """
    bins = 20  # bin value
    # plotting the histogram for the best feature
    histogram, bins, patches = plt.hist(dataframe['Standard Deviation'], bins, edgecolor='blue')
    plt.axvline(best_threshold, color='red', linestyle='--')  # marking the best threshold
    # looping through the bins and setting the color black for bins[i] which is greater than threshold or else green
    for i in range(len(bins) - 1):
        if bins[i] > best_threshold:
            patches[i].set_facecolor('black')
        else:
            patches[i].set_facecolor('green')
    plt.xlabel('Feature')
    plt.ylabel('Threshold')
    plt.title('Feature')
    plt.savefig('Histogram for best feature.png')


def one_rule(best_feature, best_threshold):
    """
    Printing the one rule
    :param best_feature: best feature
    :param best_threshold: best threshold
    :return: code pipeline
    """
    code_pipeline = f"""
    if {best_feature} <= {best_threshold}:
        class = lawn
    else:
        class = other
    """

    return code_pipeline


def main():
    """
    Main method
    :return: None
    """
    parser = argparse.ArgumentParser(description='Feature Selection')
    parser.add_argument('filepath', type=Path)
    args = parser.parse_args()

    data = processing(args.filepath)
    computed_dataframe = computation(data)
    best_feature, best_threshold = feature_selection_and_threshold(computed_dataframe)
    plot(computed_dataframe, best_threshold)
    print(one_rule(best_feature, best_threshold))
    print("The best feature is ", best_feature)
    print("The best threshold is ", best_threshold)


if __name__ == '__main__':
    main()
