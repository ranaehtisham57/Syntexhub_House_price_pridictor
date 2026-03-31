import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def main():
    print("Loading Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Generate CSV
    csv_path = 'iris.csv'
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to '{csv_path}'")
    
    # Create an images directory
    os.makedirs('images', exist_ok=True)
    
    print("Performing EDA...")
    # Visualize feature pairs
    sns.pairplot(df, hue='species', vars=iris.feature_names)
    plt.savefig('images/pairplot.png')
    plt.close()
    print("Saved pairplot to 'images/pairplot.png'")
    
    # Train-test split
    X = df[iris.feature_names]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Classifiers...")
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Accuracy:       {acc_dt:.4f}")
    
    # Choose best model to save
    best_model = lr if acc_lr >= acc_dt else dt
    model_name = "Logistic Regression" if acc_lr >= acc_dt else "Decision Tree"
    print(f"\nBest model: {model_name}. Saving to 'flower_model.pkl'...")
    joblib.dump(best_model, 'flower_model.pkl')
    
    # Confusion Matrix for best model
    best_y_pred = y_pred_lr if acc_lr >= acc_dt else y_pred_dt
    cm = confusion_matrix(y_test, best_y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig('images/confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix to 'images/confusion_matrix.png'")
    
    # Identify misclassifications for output interpretation
    print("\nInterpretation of Misclassifications:")
    misclassified = X_test[y_test != best_y_pred]
    if len(misclassified) > 0:
        print(f"Found {len(misclassified)} misclassified samples.")
        for idx in misclassified.index:
            actual = iris.target_names[y_test[X_test.index == idx][0]]
            predicted = iris.target_names[best_y_pred[list(X_test.index).index(idx)]]
            print(f"Index {idx} | Actual: {actual} -> Predicted: {predicted}")
            print(f"Features: {dict(X_test.loc[idx])}")
    else:
        print("No misclassifications found! Perfect accuracy on test set.")
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_y_pred, target_names=iris.target_names))
    
    print("\nModel training and evaluation complete.")

if __name__ == '__main__':
    main()
