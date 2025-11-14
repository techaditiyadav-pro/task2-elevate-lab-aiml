import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda():
    df = pd.read_csv("data/titanic.csv")
    
    print("\n---- Data Head ----")
    print(df.head())

    print("\n---- Summary Statistics ----")
    print(df.describe())

    print("\n---- Missing Values ----")
    print(df.isnull().sum())

    # Visuals Folder Plots
    sns.histplot(df['Age'], kde=True)
    plt.title("Age Histogram")
    plt.savefig("visuals/histogram_age.png")
    plt.clf()

    sns.boxplot(x=df['Fare'])
    plt.title("Fare Boxplot")
    plt.savefig("visuals/boxplot_fare.png")
    plt.clf()

    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("visuals/correlation_heatmap.png")
    plt.clf()

if __name__ == "__main__":
    run_eda()
