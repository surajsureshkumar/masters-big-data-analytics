"""
Author - Suraj Sureshkumar ( ss7495)
HomeWork - 02
Description - Basics One Dimensional Classification
"""

import os
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def processing(master_dataframe):
    """
    reading all the station folders and looping through each folder and getting each csv file and adding them to
    the dataframe
    :param master_dataframe:The data frame with all data
    :return:
    """
    root_directory = "CS720_TRAFFIC_STATIONS"  # root directory path
    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            master_dataframe[0] = pd.concat([master_dataframe[0], data])  # concating the data from csv files
    master_dataframe[0].reset_index(drop=True, inplace=True)
    master_dataframe[0] = master_dataframe[0].round(0)  # rounding the speeds
    bins = np.arange(0, 101, 1.0)  # creating bins
    # grouping the speeds with bins
    master_dataframe[0]['binned_speeds'] = master_dataframe[0][['SPEED']].apply(lambda x: pd.cut(x, bins))

    return master_dataframe[0]  # returning the data frame


def manipulation(length, df, lock):
    """
    Makes a copy of the data frame
    :param df: data frame
    :return: none
    """
    temp_df = df.copy()


def summary_histogram(dataframe):
    """
    This function plots the summary histogram
    :param dataframe: is the dataframe containing all the data
    :return: none
    """
    bins = np.arange(40, 101, 1.0)  # bins
    plt.hist(dataframe['SPEED'], bins=bins, edgecolor='black')  # plotting the histogram
    plt.xlabel("Speeds")
    plt.ylabel('People')
    plt.title('Summary of data from all threads')
    plt.savefig('Summary Histogram.png')  # saving the figure as png


def people_not_trying_to_speed(dataframe):
    """
    Plots the histogram of people not trying to speed
    :param dataframe: data frame with all the data
    :return: none
    """
    bins = np.arange(40, 101, 1.0)  # bins
    people_who_are_not_speeding = dataframe[dataframe['INTENT'] != 2]  # takes the data of where Intent = 0 and 1
    plt.figure()
    plt.xlabel("Speeds (mph)")
    plt.ylabel('Not trying to speed')
    plt.title('People not trying to speed')
    plt.hist(people_who_are_not_speeding['SPEED'], bins, color="green", edgecolor='black')  # plotting the histogram
    plt.savefig('people not trying to speed.png')  # saving the plot as a png file


def people_trying_to_speed(dataframe):
    """
    Plots the histogram for people trying to speed
    :param dataframe: dataframe containing all the data
    :return: none
    """
    bins = np.arange(40, 101, 1)  # bins
    people_trying_to_speeding = dataframe[dataframe['INTENT'] == 2]  # Takes the data of people who are speeding
    plt.figure()
    plt.xlabel("Speeds (mph)")
    plt.ylabel('Trying to speed')
    plt.title('People trying to speed')
    plt.hist(people_trying_to_speeding['SPEED'], bins, color="red", edgecolor='black')  # plotting the histogram
    plt.savefig("people trying to speed.png")  # saving the plot as png file


def find_threshold(dataframe):
    """
    This function finds the threshold
    :param dataframe: dataframe containing all the data
    :return: best_threshold, minimum_errors, best_false_positive_rate, best_true_positive_rate,
    false_positive_rate_list , true_positive_rate_list
    """
    best_threshold = 999999  # initializing the best threshold to 9999 later for comparing
    minimum_errors = float('inf')  # minimum error is initialized to infinity
    sorted_speeds = sorted(set(dataframe['SPEED']))  # sorting the speeds
    non_speeders = dataframe[dataframe['INTENT'] != 2]  # takes the data of people whose intent are 0 and 1
    speeders = dataframe[dataframe['INTENT'] == 2]  # speeders stores the data of people who are aggresive drivers

    # lists to store the false positive and true positive rate values
    false_positive_rate_list = []
    true_positive_rate_list = []

    # initializing false and true positive rate to 0
    best_false_positive_rate = 0
    best_true_positive_rate = 0

    # looping through sorted speeds
    for possible_threshold_speed in sorted_speeds:
        # storing the false alarms where we compare it to the possible threshold speed which is greater than and taking
        # the whole length of it
        false_alarms = len(non_speeders.loc[non_speeders['SPEED'] > possible_threshold_speed])
        # storing the false negatives where we compare it to the possible threshold speed which is less than or equal to
        # and taking the whole length of it
        false_negatives = len(speeders.loc[speeders['SPEED'] <= possible_threshold_speed])

        # finding the true alarms
        true_alarms = len(speeders) - false_negatives

        # finding the false positive and true positive rates
        false_positive_rate = false_alarms / len(non_speeders)
        true_positive_rate = true_alarms / len(speeders)

        # appending the values of false positive and true positive rate to their respective lists
        false_positive_rate_list.append(false_positive_rate)
        true_positive_rate_list.append(true_positive_rate)

        # finding the total number of mistakes
        total_number_of_mistakes = false_alarms + false_negatives

        # comparing to find the best threshold
        # comparing the total number of mistakes with minimum errors
        if total_number_of_mistakes <= minimum_errors:
            # setting the best threshold as the possible threshold speed
            # minimum errors takes the value of total number of mistakes
            # setting the best false and true positive rates
            best_threshold = possible_threshold_speed
            minimum_errors = total_number_of_mistakes
            best_false_positive_rate = false_positive_rate
            best_true_positive_rate = true_positive_rate
    # returning all the values
    return best_threshold, minimum_errors, best_false_positive_rate, best_true_positive_rate, false_positive_rate_list \
        , true_positive_rate_list


def classifier(threshold):
    """
    The classifier program
    :param threshold: the threshold value found from the above function
    :return: none
    """
    # this code pipeline hold the code that needs to be written in the classifier file
    code_pipeline = f"""
def classifier(speed):
    if speed < {threshold}:
        intent = 1
    else:
        intent = 2
    return intent
"""
    # opening the file and writing the code_pipeline into it
    with open("HW_02_Sureshkumar_Suraj_Classifier.py", 'w') as file:
        file.write(code_pipeline)


def roc_curve(false_positive_rate_list, true_positive_rate_list, false_positive_rate, true_positive_rate):
    """
    Plotting the ROC Curve
    :param false_positive_rate_list: all values of false positive
    :param true_positive_rate_list: all values of true positive
    :param false_positive_rate: value of the false_positive_rate
    :param true_positive_rate: value of the true_positive_rate
    :return: none
    """
    plt.figure(figsize=(10, 6))  # figure size
    plt.plot(false_positive_rate_list, true_positive_rate_list, color='blue')  # plotting the curve

    plt.xlabel("False Positive Rate")
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.scatter(false_positive_rate_list, true_positive_rate_list)  # scatter plot
    plt.scatter(false_positive_rate, true_positive_rate, color=[0, 0.8, 0], s=150)
    plt.savefig("ROC Curve.png")  # saving the plot


def main():
    """
    Main Method
    :return: Nothing
    """
    master_dataframe = [pd.DataFrame(columns=['SPEED', 'INTENT'])]  # dataframe
    data_frame = processing(master_dataframe)  # sending the dataframe to processing
    df_len = len(data_frame)  # length of the data frame

    lock = threading.Lock()  # using a lock

    # below are the four threads where each thread splits the data equally and takes a lock for each thread
    # which provides the exclusive access
    t1 = threading.Thread(target=manipulation, args=(data_frame[:df_len // 4], master_dataframe, lock))
    t2 = threading.Thread(target=manipulation,
                          args=(data_frame[df_len // 4: (df_len // 4) * 2], master_dataframe, lock))
    t3 = threading.Thread(target=manipulation,
                          args=(data_frame[(df_len // 4) * 2: (df_len // 4) * 3], master_dataframe, lock))
    t4 = threading.Thread(target=manipulation, args=(data_frame[(df_len // 4) * 3:], master_dataframe, lock))

    # starting the threads
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    # joining all the threads
    t1.join()
    t2.join()
    t3.join()
    t4.join()

    # calling the functions to plot the histogram
    summary_histogram(master_dataframe[0])
    people_not_trying_to_speed(master_dataframe[0])
    people_trying_to_speed(master_dataframe[0])

    # all the values returned by the find_threshold function which stores the respective values
    best_threshold, minimum_errors, best_false_positive_rate, best_true_positive_rate, false_positive_rate_list \
        , true_positive_rate_list = find_threshold(master_dataframe[0])

    # printing the best threshold,best false positive and best true positive rates
    print("The best threshold for speed = ", best_threshold)
    print("The best false alarm rate is = ", best_false_positive_rate)
    print("The best true positive rate is = ", best_true_positive_rate)

    classifier(best_threshold)  # calling the classifier function

    # calling the roc_curve function to plot the curve and passing the necessary arguments
    roc_curve(false_positive_rate_list, true_positive_rate_list, best_false_positive_rate, best_true_positive_rate)


if __name__ == "__main__":
    main()
