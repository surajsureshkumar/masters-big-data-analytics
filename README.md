# README.md
# masters-big-data-analytics
## Vehicular Speed Clustering Using Otsu's Method

### Description
This repository contains implementations of various big data analytics assignments. Every directory can be cloned and run locally and code is documented and easy to understand but a bit knowledge of the concepts is required.

### Prerequisites
Ensure that the following software and libraries are installed:
- Python 3.7 or higher
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
