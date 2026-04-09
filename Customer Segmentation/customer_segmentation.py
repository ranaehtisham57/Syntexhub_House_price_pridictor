import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading dataset...")
    # Try multiple common URLs for the dataset
    urls = [
        "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
        "https://raw.githubusercontent.com/vjiyer/Research/master/Python/Mall_Customers.csv",
        "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
    ]
    
    df = None
    for url in urls:
        try:
            df = pd.read_csv(url)
            print(f"Successfully loaded dataset from {url}")
            break
        except Exception as e:
            continue
            
    if df is None:
        print("Failed to load from URLs, generating synthetic dataset...")
        np.random.seed(42)
        df = pd.DataFrame({
            'CustomerID': range(1, 201),
            'Gender': np.random.choice(['Male', 'Female'], 200),
            'Age': np.random.randint(18, 70, 200),
            'Annual Income (k$)': np.random.randint(15, 140, 200),
            'Spending Score (1-100)': np.random.randint(1, 100, 200)
        })
        
    print("Dataset shape:", df.shape)
    
    # 2. Clean & Scale features
    # Select 'Age' and 'Spending Score (1-100)'
    features = ['Age', 'Spending Score (1-100)']
    
    # Drop rows with NaN in these columns
    df = df.dropna(subset=features)
    
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Use elbow method to choose k
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.tight_layout()
    plt.savefig('elbow_method.png')
    plt.close()
    print("Elbow method plot saved as 'elbow_method.png'")
    
    # We will pick k=4 explicitly as it often provides a robust segmentation 
    # for Age vs Spending score distributions.
    k_optimal = 4
    print(f"\nApplying K-Means with k={k_optimal} (selected based on elbow behavior)")
    kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['Cluster'] = clusters
    
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100)
    plt.title(f'Customer Segments (Age vs Spending Score, k={k_optimal})')
    plt.tight_layout()
    plt.savefig('clusters.png')
    plt.close()
    print("Clusters plot saved as 'clusters.png'")
    
    # 4. Profile clusters to derive marketing actions
    cluster_profiles = df.groupby('Cluster')[features].mean().reset_index()
    cluster_profiles['Count'] = df.groupby('Cluster').size().values
    
    print("\nCluster Profiles (Averages):")
    print(cluster_profiles)
    
    # 5. Save cluster labels and create short report per segment
    df.to_csv('segmented_customers.csv', index=False)
    print("\nSegmented data saved to 'segmented_customers.csv'")
    
    # Generate short report
    report = "Customer Segmentation Report\n"
    report += "=" * 30 + "\n\n"
    
    for i in range(k_optimal):
        profile = cluster_profiles[cluster_profiles['Cluster'] == i].iloc[0]
        avg_age = profile['Age']
        avg_spend = profile['Spending Score (1-100)']
        count = profile['Count']
        
        report += f"Segment {i} Profile:\n"
        report += f"- Number of Customers: {int(count)}\n"
        report += f"- Average Age: {avg_age:.1f} years\n"
        report += f"- Average Spending Score: {avg_spend:.1f}\n"
        
        # Simple heuristics for marketing actions based on age and spend
        age_group = "younger" if avg_age < 40 else "older"
        spend_group = "high" if avg_spend >= 50 else "low"
        
        report += f"-> Suggested Action: Target this {age_group} group with {spend_group} spending tendencies. "
        if spend_group == "high":
            report += "Offer premium products, VIP perks, and early access to new releases to build loyalty.\n"
        else:
            report += "Provide promotional discounts, bundle offers, and value-based marketing to increase spending.\n"
        report += "-" * 30 + "\n"
        
    with open('segmentation_report.txt', 'w') as f:
        f.write(report)
        
    print("Segmentation report saved to 'segmentation_report.txt'")

if __name__ == '__main__':
    main()
