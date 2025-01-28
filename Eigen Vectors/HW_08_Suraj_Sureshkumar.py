import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def data_processing():
    """
    Reading the data from csv file and creating a  dataframe
    :return: dataframe
    """
    dataframe = pd.read_csv('HW_CLUSTERING_SHOPPING_CART_v2231a.csv')
    dataframe = dataframe.drop(columns=['ID'])
    return dataframe


def covariance_matrix_and_eigen_vectors(dataframe):
    """
    Calculating the covariance matrix and eigen vectors
    :param dataframe: the data points
    :return: covariance matrix, eigen values and eigen vectors
    """
    covariance = dataframe.cov()
    eigen_values, eigen_vectors = np.linalg.eig(covariance)
    return covariance, eigen_values, eigen_vectors.round(1).T


def normalized_eigen_values(eigen_values):
    """
    Normalizing the eigen values
    :param eigen_values: the eigen values
    :return: cumulative sum of normalized eigen values
    """
    sorted_values = sorted(abs(eigen_values))
    summed_absolute_values = np.sum(np.abs(sorted_values))
    normalized_values = sorted_values / summed_absolute_values
    cumulative_sum = np.cumsum(normalized_values)

    return cumulative_sum


def plot_eigen_value(cumulative_sum):
    """
    Plotting the cumulative sum of the eigen values
    :param cumulative_sum: the cumulative sum of the eigen values
    :return: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_sum, marker='o')
    plt.xlabel('Eigenvalues')
    plt.ylabel('Normalized eigen values')
    plt.title('Cumulative sum of normalized eigen values')
    plt.savefig('Normalized Eigenvalues.png')
    plt.show()


def first_three_eigen_vectors(eigen_values, eigen_vectors):
    """
    Getting the top three eigen vectors
    :param eigen_values: the eigen values
    :param eigen_vectors: eigen vectors
    :return: the top three eigen vectors
    """
    eigen_values = list(zip(abs(eigen_values), eigen_vectors))
    sorted_eigen_values = sorted(eigen_values, reverse=True)
    return np.array([x[1] for x in sorted_eigen_values[:3]])


def scatter_plot(projections):
    """
    Scatter plot for the two eigen vectors
    :param projections: the projections of the data points on the eigen vectors
    :return: None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(projections['A'], projections['B'])
    plt.xlabel('First eigen vector points ')
    plt.ylabel('Second eigen vector points ')
    plt.title('Scatter plot of first two eigen vectors')
    plt.savefig('First two eigen vectors.png')
    plt.show()


def pca(dataframe, eigen_vectors):
    """
    Performing pca and kmeans
    :param dataframe: the data points
    :param eigen_vectors:  the eigen vectors
    :return: None
    """
    eigen_vector1 = eigen_vectors[0]  # getting the first vector
    eigen_vector2 = eigen_vectors[1]  # getting the second vector
    vectors = np.array([eigen_vector1, eigen_vector2]).T  # creating a matrix of the two vectors
    dot_product = np.dot(dataframe, vectors)  # dot product
    dot_product = pd.DataFrame(dot_product, columns=['A', 'B'])
    scatter_plot(dot_product)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(dot_product)
    print('The cluster centers are', kmeans.cluster_centers_)
    original_cluster_center = np.dot(kmeans.cluster_centers_, vectors.T).round(1)
    print('The cluster centers in original space are', original_cluster_center)


def main():
    """
    Main method
    :return: None
    """
    dataframe = data_processing()
    covariance_matrix, eigen_values, eigen_vectors = covariance_matrix_and_eigen_vectors(dataframe)
    cumulative_sum_of_eigen_values = normalized_eigen_values(eigen_values)
    plot_eigen_value(cumulative_sum_of_eigen_values)
    top_eigen_vectors = first_three_eigen_vectors(eigen_values, eigen_vectors)
    print("The top three eigen vectors are", top_eigen_vectors)  # printing the top three vectors
    pca(dataframe, top_eigen_vectors[:2])


if __name__ == '__main__':
    main()
