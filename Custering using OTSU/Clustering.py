"""
Author: Suraj Sureshkumar(ss7495)
Homework 1
Description: implementation of the otsu method to separate vehicular data into two clusters
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def data_process_and_binning():
    """
    reading all the station folders and looping through each folder and getting each csv file and adding them to
    the dataframe
    :return: a dataframe
    """
    root_directory = "D:\BDA_HW1\TRAFFIC_STATIONS_ALL"
    master_dataframe = pd.DataFrame()

    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path, header=None)
            master_dataframe = master_dataframe._append(data, ignore_index=True)

    bins = pd.interval_range(start=40, end=80, freq=0.5)  # creating bins
    master_dataframe['binned_speeds'] = master_dataframe.apply(lambda x: pd.cut(x, bins))

    return master_dataframe


def otsu(data_points):
    """
    Otsu method
    :param data_points: data frame from the data_process_and_binning()
    :return:
    """
    best_mixed_variance = float('inf')  # initially infinity
    best_threshold = 0
    best_threshold_2 = 0
    variance = []
    total = []
    norm_factor = 30
    best_cost_function = float('inf')  # initially infinity
    return_set = []

    # looping through 40 to 80 in increments of 0.5 and appending each value to the total
    for threshold in np.arange(40, 80, 0.5):
        total.append(threshold)
        # if the type is not of pd.series then add it to the leftset for the values less than threshold and to the right
        # set for the values greater than threshold
        if type(data_points) != pd.Series:
            left_set = data_points[0][data_points[0] <= threshold]
            right_set = data_points[0][data_points[0] > threshold]
        else:
            left_set = data_points[data_points <= threshold]
            right_set = data_points[data_points > threshold]

        # computing the wt_left and wt_right
        wt_left = len(left_set) / len(data_points)
        wt_right = len(right_set) / len(data_points)

        left_set = pd.Series(list(left_set))
        right_set = pd.Series(list(right_set))

        # the below two lines of code finds the variance for the left and right set
        variance_left = left_set.var()
        variance_right = right_set.var()

        # calculating the mixed variance
        mixed_variance = (wt_left * variance_left) + (wt_right * variance_right)
        variance.append(mixed_variance)

        # comparing the mixed variance with the best_mixed_variance and setting the new best_mixed_variance
        # storing the left in return_set to run otsu on the left_set again
        # here return_set stores the left_set
        if mixed_variance < best_mixed_variance:
            best_mixed_variance = mixed_variance
            best_threshold = threshold
            return_set = left_set

        # calculating the regularization
        regularization = abs((len(left_set) - len(right_set)) / norm_factor)

        # calculating the cost function
        cost_function = mixed_variance + regularization

        # finding the best_threshold_2 by first comparing the cost_function with the
        # best_cost_function(initially infinity) and then setting the best_threshold_2 as threshold
        if cost_function < best_cost_function:
            best_cost_function = cost_function
            best_threshold_2 = threshold

    return best_threshold, variance, total, best_cost_function, return_set, best_threshold_2


def hist_plot(threshold_1, data, threshold_2=None):
    """
    Plotting the histogram
    :param threshold_1: initial threshold
    :param data: the data points
    :param threshold_2: initially set to None, the threshold returned by the otsu function
    :return:
    """
    bins = np.arange(40, 81, 0.5)  # creating bins
    histogram, bins, patches = plt.hist(data[0], bins, edgecolor='black')  # plotting the histogram
    plt.xlim(40, 80)
    # below two lines denote teh x and y label
    plt.xlabel("Speeds (mph)")
    plt.ylabel('Number of vehicles')
    # looping through the number of bins and if bins[i] less than threshold_2 then we paint it red or else
    # if the bins[i] lesser than threshold one which is 62.5 then it will be colored blue till 61.6 and the rest will
    # colored green
    for i in range(len(bins) - 1):
        if threshold_2 != None and bins[i] <= threshold_2:
            patches[i].set_facecolor('red')
        elif bins[i] <= threshold_1:
            patches[i].set_facecolor('blue')
        else:
            patches[i].set_facecolor('green')
    plt.show()


def plot(best_threshold, variance, total):
    """
    Plotting the mixed_variance vs speed curve and denoting the threshold with green line
    :param best_threshold: is the best possible threshold to separate the two data
    :param variance: is the calculated variance from otsu()
    :param total: contains all the speeds
    :return: None
    """
    plt.figure()
    plt.plot(total, variance)
    plt.xlabel("Speeds (mph)")
    plt.ylabel('mixed variance')
    plt.axvline(x=best_threshold, color='green', linestyle='--', label=f'Best threshold at {best_threshold:.2f} mph')
    plt.show()


def main():
    """
    Main method
    :return: None
    """
    data = data_process_and_binning()  # data merging and binning
    threshold_1, variance, total, cost_function, return_set, best_threshold_2 = otsu(data)  # 1st otsu call
    hist_plot(threshold_1, data)
    print("Threshold without regularization", threshold_1)  # printing the threshold1
    print("Cost function is", cost_function)  # printing the cost function
    plot(threshold_1, variance, total)
    # below line is the 2nd otsu call on the set with the higher variance
    threshold_2, variance, total, cost_function, new_left_set, best_threshold_2 = otsu(return_set)
    # printing the threshold2 that resulted, the threshold uses regularization so the value differs a bit
    print("Threshold after regularization", best_threshold_2)
    hist_plot(threshold_1, data, threshold_2)


if __name__ == "__main__":
    main()
