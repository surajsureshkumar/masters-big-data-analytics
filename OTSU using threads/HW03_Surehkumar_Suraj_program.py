"""
Author - Suraj Sureshkumar(ss7495)
Homework 3
Description - Otsu using threads
"""
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def processing():
    """
    Reading the csv file and creating a dataframe
    :return: last column of threshold
    """
    dataframe = pd.read_csv('Lawn_Data_FOR_CLUSTERING__v20.csv')
    return dataframe.iloc[:, -1:]


def otsu(data_points, bins):
    """
    Otsu method
    :param quantization_value: the best bin value
    :param data_points: data frame from the data_process_and_binning()
    :return: returns the threshold, variance, total thresholds,and the total_running_time_without_threads
    """
    best_mixed_variance = float('inf')  # initially infinity
    best_threshold = 0
    variance = []
    total = []
    data_points = data_points.iloc[:, -1:]
    data_points = data_points.values.flatten()
    # looping through each bin and appending them to the total list

    for threshold in bins:
        total.append(threshold)
        # if the type of data points is not series then sort the data by comparing them with threshold and
        # if value is less than or equal to threshold then add it to the left set or else the right set
        if type(data_points) != pd.Series:
            left_set = data_points[data_points <= threshold]
            right_set = data_points[data_points > threshold]

        wt_left = len(left_set) / len(data_points)
        wt_right = len(right_set) / len(data_points)

        # left_set = pd.Series(list(left_set))
        # right_set = pd.Series(list(right_set))

        # the below two lines of code finds the variance for the left and right set
        variance_left = np.var(left_set)
        variance_right = np.var(right_set)

        # calculating the mixed variance
        mixed_variance = (wt_left * variance_left) + (wt_right * variance_right)
        variance.append(mixed_variance)

        # comparing the mixed variance with the best_mixed_variance and setting the new best_mixed_variance
        # storing the left in return_set to run otsu on the left_set again
        if mixed_variance < best_mixed_variance:
            best_mixed_variance = mixed_variance
            best_threshold = threshold

    return best_threshold, variance, total, best_mixed_variance


def hist_quantize(data):
    """
    Drawing the histogram for different quantization values
    :param data: data points
    :return: a dictionary
    """
    hist_values = {}  # dictionary
    quantization_values = [1, 2 / 3, 1 / 2, 1 / 4, 1 / 5, 1 / 6, 1 / 8, 1 / 16, 1 / 32, 1 / 40, 1 / 50, 1 / 64]
    data_flattened = data.values.flatten()

    # looping through the quantization values and plotting the histogram
    for values in quantization_values:
        bins = np.arange(3.0, 10.0, values)  # creating bins

        # np.histogram counts the frequency of the bins
        histogram, bin_edges = np.histogram(data_flattened, bins)
        hist_values[values] = histogram

        # plotting the actual histogram
        plt.hist(data_flattened, bins, edgecolor='black')
        hist_values[values] = histogram  # adding the values to the dictionary

        plt.title(f'Data Quantized into Bins of Size {values} inches')
        plt.xlabel('Plant Length')
        plt.ylabel('Frequency or Counts')
        plt.savefig(f'Data Quantized into Bins of Size {values} inches.png')  # saving the figure
        plt.clf()

    return hist_values


def cost_function(hist_values):
    """
    Cost function to find the best quantization value
    :param hist_values: dictionary holding the count
    :return: quantized value
    """
    best_cost_function = float('inf')  # initially infinity
    best_quantization_value = 0

    # looping through the dictionary
    for key, values in hist_values.items():
        total_difference = 0
        for i in range(1, len(values) - 1):
            # finding the average difference and the total difference
            average_difference_between_neighbors = (values[i - 1] + values[i + 1]) / 2
            total_difference += abs(values[i] - average_difference_between_neighbors)

        average_difference = total_difference / (len(values) - 2)

        # finding the best cost function and the quantization value
        if average_difference < best_cost_function:
            best_cost_function = average_difference
            best_quantization_value = key

    # truncating the value to upto 4 decimal places
    quantization_truncated = np.floor(best_quantization_value * 10000) / 10000

    return quantization_truncated


def plot(best_threshold, variance, total,best_mixed_variance):
    """
    Plotting the mixed_variance vs speed curve and denoting the threshold
    :param best_threshold: is the best possible threshold to separate the two data
    :param variance: is the calculated variance from otsu()
    :param total: contains all the speeds
    :return: None
    """
    plt.figure(figsize=(10, 6))  # figure size
    plt.plot(total, variance, color='blue')  # plotting
    plt.xlabel("mixed variance")
    plt.ylabel('threshold')
    plt.title('mixed variance vs threshold')
    plt.scatter(total, variance)  # scatter plot
    plt.scatter(best_threshold, best_mixed_variance, color=[0, 0.8, 0], s=150)
    plt.savefig(f'Mixed variance vs the threshold.png')
    plt.clf()


def otsu_helper(data, bins, thread_number, output):
    """
    Helper function
    :param data: data points
    :param bins: bins to store data points
    :param thread_number: the number of the thread
    :param output: list containing the threshold and variance
    :return: none
    """
    threshold, variance, all_thresholds, best_mixed_variance = otsu(data, bins)
    output[thread_number] = (threshold, best_mixed_variance)


def main():
    """
    Main method
    :return: None
    """
    data = processing()  # processing the data

    hist_values = hist_quantize(data)  # storing the frequency in dictionary
    quantization_value = cost_function(hist_values)
    print("Best cost function to bin the data", quantization_value)

    # running otsu without the threads
    without_thread_start_time = time.time()
    bins = np.arange(3.0, 10.0, quantization_value)
    threshold_1, variance, all_thresholds, best_mixed_variance = otsu(data, bins)

    without_thread_end_time = time.time()
    print("Threshold is ", threshold_1)
    print("Mixed variance is ", best_mixed_variance)
    print('Time without threads is ', without_thread_end_time - without_thread_start_time)
    plot(threshold_1, variance, all_thresholds,best_mixed_variance)  # plotting the histogram

    # creating the bins and taking its legnth to segment the bins
    bins = np.arange(2.0, 10.0, quantization_value)
    num_bins = len(bins)

    # starting the timer
    threading_start_time = time.time()
    output = [None] * 3  # output result
    # three threads
    t1 = threading.Thread(target=otsu_helper, args=(data, bins[:num_bins // 3], 0, output))
    t2 = threading.Thread(target=otsu_helper, args=(data, bins[num_bins // 3: (num_bins // 3) * 2], 1, output))
    t3 = threading.Thread(target=otsu_helper, args=(data, bins[(num_bins // 3) * 2:], 2, output))

    # starting the threads
    t1.start()
    t2.start()
    t3.start()
    # joining the threads
    t1.join()
    t2.join()
    t3.join()

    # find the best threshold after using threads on otsu
    best_threshold = 0
    best_cost = float('inf')
    # looping through the output list and finding the best cost and the best threshold
    for threshold, variance in output:
        if variance < best_cost:
            best_cost = variance
            best_threshold = threshold
    print("Best threshold after threading ", best_threshold)
    threading_end_time = time.time()
    total_time = threading_end_time - threading_start_time

    print('Using Threads', total_time)


if __name__ == "__main__":
    main()
