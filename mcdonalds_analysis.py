# -*- coding: utf-8 -*-
"""
McDonald's Market Segmentation Analysis - Cleaned and Optimized Version
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use 'Agg' backend for better compatibility (non-GUI / script execution)
matplotlib.use('Agg')  # Change to 'TkAgg' if interactive GUI is needed

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde


class McDonaldAnalysis:
    def __init__(self, data_path="mcdonalds.csv"):
        self.data_path = data_path
        self.mcdonalds = None
        self.MD_x = None
        self.MD_pca = None
        self.MD_km28 = {}
        
    def load_data(self):
        """Load and preprocess the data"""
        print("Loading data...")
        try:
            self.mcdonalds = pd.read_csv(self.data_path)
            self.MD_x = (self.mcdonalds.iloc[:, :11] == "Yes").astype(int).values
            
            like_map = {"I hate it!-5": -5, "-4": -4, "-3": -3, "-2": -2, "-1": -1,
                        "0": 0, "+1": 1, "+2": 2, "+3": 3, "+4": 4, "I love it!+5": 5}
            visit_map = {"Never": 1, "Once a year": 2, "Every three months": 3,
                         "Once a month": 4, "Once a week": 5, "More than once a week": 6}
            self.mcdonalds['Like.n'] = self.mcdonalds['Like'].map(like_map)
            self.mcdonalds['VisitNumeric'] = self.mcdonalds['VisitFrequency'].map(visit_map)
            
            print(f"Loaded {self.mcdonalds.shape[0]} rows and {self.mcdonalds.shape[1]} columns.")
            return True
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False

    def show_data_info(self):
        print("\n=== Data Snapshot ===")
        print(self.mcdonalds.head(3))
        print("\nBinary Attribute Means:")
        for col, val in zip(self.mcdonalds.columns[:11], self.MD_x.mean(axis=0)):
            print(f"{col:12s}: {val:.2f}")
    
    def perform_pca(self):
        print("\n=== Principal Component Analysis ===")
        self.MD_pca = PCA()
        self.MD_pca.fit(self.MD_x)
        
        summary = pd.DataFrame({
            'Standard deviation': np.sqrt(self.MD_pca.explained_variance_),
            'Proportion of Variance': self.MD_pca.explained_variance_ratio_,
            'Cumulative Proportion': np.cumsum(self.MD_pca.explained_variance_ratio_)
        }, index=[f'PC{i+1}' for i in range(len(self.MD_pca.explained_variance_))])
        
        print(summary.head(5).to_string(float_format="%.4f"))
        self.plot_pca_results()

    def plot_pca_results(self):
        try:
            scores = self.MD_pca.transform(self.MD_x)
            loadings = self.MD_pca.components_.T[:, :2]
            plt.figure(figsize=(10, 8))
            plt.scatter(scores[:, 0], scores[:, 1], color='grey', alpha=0.6)

            scale = np.max(np.abs(scores[:, :2])) / np.max(np.abs(loadings)) * 0.7
            for i, (x, y) in enumerate(loadings * scale):
                plt.arrow(0, 0, x, y, color='red', head_width=0.1)
                plt.text(x*1.1, y*1.1, self.mcdonalds.columns[i], fontsize=9)

            plt.axhline(0, color='black', linestyle='--', alpha=0.3)
            plt.axvline(0, color='black', linestyle='--', alpha=0.3)
            plt.xlabel(f"PC1 ({self.MD_pca.explained_variance_ratio_[0]*100:.1f}% var)")
            plt.ylabel(f"PC2 ({self.MD_pca.explained_variance_ratio_[1]*100:.1f}% var)")
            plt.title("PCA Biplot")
            plt.grid(alpha=0.3)
            plt.savefig("pca_biplot.png")  # Save for review
            plt.close()
        except Exception as e:
            print(f"Failed to plot PCA: {e}")

    def perform_kmeans(self, k_values=range(2, 9)):
        print("\n=== Running K-Means ===")
        for k in k_values:
            km = KMeans(n_clusters=k, n_init=10, random_state=123)
            km.fit(self.MD_x)
            self.MD_km28[k] = km
            print(f"k={k}: Cluster sizes = {np.bincount(km.labels_)}")
        self.plot_scree()

    def plot_scree(self):
        try:
            inertias = [self.MD_km28[k].inertia_ for k in sorted(self.MD_km28)]
            plt.figure(figsize=(8, 5))
            plt.plot(list(self.MD_km28), inertias, 'bo-')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Inertia")
            plt.title("Scree Plot")
            plt.grid(True)
            plt.savefig("kmeans_scree_plot.png")
            plt.close()
        except Exception as e:
            print(f"Failed to plot scree: {e}")

    def analyze_segments(self, k=4):
        if k not in self.MD_km28:
            print(f"Run perform_kmeans() before analyzing {k}-segments.")
            return
        print(f"\n=== Segment Analysis for k={k} ===")
        labels = self.MD_km28[k].labels_
        centers = self.MD_km28[k].cluster_centers_
        # You can call more functions here like plot_segment_profiles if needed

    def main_workflow(self):
        if not self.load_data():
            return
        self.show_data_info()
        input("\nPress Enter to continue with PCA...")
        self.perform_pca()
        input("\nPress Enter to continue with K-Means...")
        self.perform_kmeans()
        input("\nPress Enter to analyze segments (k=4)...")
        self.analyze_segments(k=4)
        print("\nAnalysis Complete.")


if __name__ == "__main__":
    McDonaldAnalysis().main_workflow()
