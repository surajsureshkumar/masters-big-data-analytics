"""
Author - Suraj Sureshkumar
Homework 5
Description - Program to demonstrate gradient descent
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process():
    """
    Loading the data into the dataframe
    :return: dataframe
    """
    dataframe = pd.read_csv('HW_05_Data.csv')
    return dataframe


def gradient_descent(dataframe):
    """
    Function that performs the gradient descent
    :param dataframe: data points
    :return:best threshold, best angle, missclassifications
    """
    # creating a points array with the x and y coordinates
    points = dataframe[["RoadDist", "ElevationChange"]].values
    output = pd.DataFrame({'TripType': dataframe['TripType']})  # output dataframe
    # lines 30 - 33 variables
    best_threshold = None
    best_mis_classifications = float('inf')
    best_angle = None
    step_size = 15
    radian_angle = np.arange(0, 90, 15)  # radian angles

    while step_size > 0.5:
        # for each angle in radian angle
        for angle in radian_angle:
            # create a vector with the current angle and project that point
            vector = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            point_projection = np.dot(points, vector)  # dot product
            min_proj = point_projection.min()  # finding the minimum projection
            max_proj = point_projection.max()  # finding the maximum projection
            # looping through the min ,max and difference of the min and max
            for thresh in np.arange(min_proj, max_proj, (max_proj - min_proj) / 100):
                # classifying the data on good and bad
                output['Classification'] = np.where(point_projection < thresh, 'Good', 'Bad')
                # calculating the misclassifications
                misclassifications = len(output[output['TripType'] != output['Classification']])
                # if the found misclassifications less than the best misclassification then the best_mis_classification
                # is the found misclassifications and set the new best ange and best thershold
                if misclassifications < best_mis_classifications:
                    best_mis_classifications = misclassifications
                    best_angle = angle
                    best_threshold = thresh
        # repeating the procedure but with half the step size
        radian_angle = np.arange(best_angle - step_size, best_angle + step_size, step_size / 2)
        step_size = step_size / 2  # reducing the step size
    return best_threshold, best_angle, best_mis_classifications


def plot_decision_boundary(best_angle, best_threshold, data):
    """
    Plotting the desicion boundary
    :param best_angle: the best angle found
    :param best_threshold: the best thershold found
    :param data: the dataframe
    :return:none
    """
    plt.figure(figsize=(10, 6))  # figure size
    # plotting the pointsbased on their x and y coordinates, contains a list of values
    plt.plot([0, best_threshold * np.cos(np.radians(best_angle))], [0, best_threshold * np.sin(np.radians(best_angle))])
    # finding the inverse slope
    decision_boundary_slope = -np.cos(np.radians(best_angle)) / np.sin(np.radians(best_angle))
    # choosing the starting point for the plot
    start_point = (best_threshold * np.cos(np.radians(best_angle)), best_threshold * np.sin(np.radians(best_angle)))
    # choosing the ending point for the plot
    end_point = (0, start_point[1] - decision_boundary_slope * start_point[0])
    # plotting the line
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], c='red', linestyle='--',
             label='A possible decision boundary')
    plt.xlim(0, 40)
    plt.ylim(0, 500)
    good_trip = data[data['TripType'] == 'Good']  # this stores the good triptype values
    bad_trip = data[data['TripType'] == 'Bad']  # this stores the bad triptype values
    # plotting the data points of good and bad trips
    plt.scatter(good_trip['RoadDist'], good_trip['ElevationChange'], c='blue', label='Good Trips')
    plt.scatter(bad_trip['RoadDist'], bad_trip['ElevationChange'], c='red', label='Bad Trips')

    plt.xlabel('Road distance (miles)')
    plt.ylabel('Change in elevation (feet)')
    plt.legend()
    plt.savefig('Gradient Descent.png')


def main():
    """
    the main method
    :return:
    """
    dataframe = process()  # getting the dataframe
    best_threshold, best_angle, missclassifications = gradient_descent(
        dataframe)  # running the gradient descent function
    # below three lines prints the best threshold, best angle and missclassifications
    print("Best threshold is ", best_threshold)
    print("Best Angle is ", best_angle)
    print("Misclassification is ", missclassifications)
    plot_decision_boundary(best_angle, best_threshold, dataframe)  # plotting the data


if __name__ == "__main__":
    main()
