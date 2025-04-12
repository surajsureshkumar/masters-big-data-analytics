# masters-big-data-analytics

### Description
This repository contains implementations of various big data analytics assignments. Every directory can be cloned and run locally and code is documented and easy to understand but a bit knowledge of the concepts is required.

### Prerequisites
Ensure that the following software and libraries are installed:
- Python 3.7 or higher.
- Required libraries listed in `requirements.txt`.

### Setup and Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/username/masters-big-data-analytics.git
   cd masters-big-data-analytics
   Adjust the directory path in the script (hw1.py) to match your dataset structure
   Run the program:
   python hw1.py

### Clustering Directory Script
The scripts inside the folders perform the following tasks:
- Processes and bins vehicular speed data from multiple CSV files.
- Includes a traffic dataset in zip format.
- Requires adjustments to the import paths based on your directory structure.
- Applies the Otsu method to determine clustering thresholds.
- Generates visualizations, including histograms with thresholds and mixed variance curves.

### One Dimensional Classification
The scripts inside the folders perform the following tasks:
- Multi-threaded data processing for efficiency.
- Identification of thresholds for aggressive and non-aggressive drivers.
- Plots summarizing driver behaviors:
  - Summary histogram.
  - Histogram of people not trying to speed.
  - Histogram of aggressive drivers.
  - ROC curve for evaluating classification performance.
 
### Decision Trees
The scripts inside the folders perform the following tasks:
- Computes statistical features (min, max, mean, std, median) from time-series data.
- Selects the best feature and threshold using a one-rule classification method.
- Minimizes misclassification errors across class labels (1, 2, 3).
- Visualizes the selected feature using a histogram with threshold marking.
- Outputs a simple if-else rule for classification based on the best feature.

### Eigen Vectors
- The scripts inside the folders perform the following tasks:
- Reads shopping cart data and performs PCA (Principal Component Analysis).
- Computes covariance matrix, eigenvalues, and eigenvectors.
- Normalizes eigenvalues and plots their cumulative sum to assess variance contribution.
- Projects data onto top two principal components for visualization.
- Applies KMeans clustering (5 clusters) on reduced data and outputs cluster centers.
- Transforms cluster centers back into the original feature space.

### Feature Selection
- The scripts inside the folders perform the following tasks:
- Implements a manually written decision tree classifier.
- Derives new features like Shagginess and ApeFactor from raw data.
- Applies nested if-else logic to classify each instance.
- Prints and stores the predicted class (1 or -1) for each data point.
- Saves the final predictions to a CSV file for validation or evaluation.

**All contributions to this repository must be original and authored by you and contribution or modifications are welcomed. Copying content or code from external sources without appropriate attribution or permission is strictly prohibited, as it may result in plagiarism and violate the repository's guidelines**
