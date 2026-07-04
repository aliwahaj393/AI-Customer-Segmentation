# `AI-Customer-Segmentation` AI-Powered Customer Segmentation Using Hierarchical Clustering

The **AI-Customer-Segmentation** project is a data science application that performs customer segmentation using machine learning techniques. By leveraging hierarchical clustering, it groups customers based on attributes such as age, profession, work experience, and spending score. This enables targeted marketing and better understanding of customer demographics.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Understanding your customer base is crucial for any business. The **AI-Customer-Segmentation** system is designed to help analysts and marketers automatically categorize customers into distinct clusters. It utilizes Agglomerative Hierarchical Clustering to identify hidden patterns within the dataset, providing actionable insights for business growth.

## Dataset

The model is trained on the [Customer Segmentation Dataset](https://www.kaggle.com/datasets/vetrirah/customer?select=Train.csv) available on Kaggle. It includes attributes such as Gender, Age, Profession, Work Experience, and Spending Score.

## Features

- Parse and clean raw customer datasets.
- Perform Exploratory Data Analysis (EDA) on customer demographics.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Execute Agglomerative Hierarchical Clustering.
- Generate visualizations including dendrograms and cluster distributions.
- Output predicted clusters for both training and testing sets.

## Getting Started

To run the **AI-Customer-Segmentation** project locally, follow these steps:

1. Clone the repository or download the source code.
2. Ensure you have Python 3.x installed.
3. Install the required libraries using pip:
   ```bash
   pip install pandas numpy scipy seaborn scikit-learn matplotlib
   ```
4. Open the Jupyter Notebook or run the Python scripts in your preferred IDE.

## Usage

1. Launch the application environment.
2. Ensure `Train.csv` and `Test.csv` are in the project directory.
3. Run the main scripts or notebook cells to:
   - Load and explore the data.
   - Preprocess the data (handle missing values, scaling, one-hot encoding).
   - Build and visualize the hierarchical clusters.
   - Export the segmented customer groups to a new CSV file.

## Notes

- The dataset should maintain the expected column structure (e.g., Age, Profession, Work_Experience, Spending_Score).
- Adjust the number of clusters in the model parameters according to your specific business needs based on the generated dendrogram.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to create a pull request.

## License

*This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)*.

## Contact

For any questions or inquiries, please contact [aliwahaj393](https://github.com/aliwahaj393).
