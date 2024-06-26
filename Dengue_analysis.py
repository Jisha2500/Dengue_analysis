# filtering and visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data

import pandas as pd

file_path = "C:/Users/jisha/Downloads/Documents/2_months.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet1')

print(data.head())


# Print the column names to check if 'P-value' exists
print("Columns in the DataFrame:", data.columns)

# Check if 'pvalue' column exists and then proceed
if 'pvalue' in data.columns:
    data['-log10(P-value)'] = -np.log10(data['P-value'])
    print(data.head())
else:
    print("Column 'P-value' does not exist in the DataFrame")


# Add a column for -log10(pvalue)
data['-log10(P-adj)'] = -np.log10(data['P-adj'])

# Filter for significant genes with log2(FC) > 2.0 and p-adj < 0.05
data['significant'] = (data['log2(FC)'] > 2.0) & (data['P-adj'] < 0.05)
data['significant'] = (data['log2(FC)'] > 2.0) & (data['P-adj'] < 0.05)

# Create a column for significance in the original data
data['significant'] = ((data['log2(FC)'] > 2) | (data['log2(FC)'] < -2)) & (data['P-adj'] < 0.05)

# Filter the data based on downregulation and upregulation
downregulated = data[data['log2(FC)'] < -2]
upregulated = data[data['log2(FC)'] > 2]

# Print downregulated genes with p-adj value
print("Downregulated Genes:")
print(downregulated[['GeneID', 'log2(FC)', 'P-adj']])

# Print upregulated genes with p-adj value
print("\nUpregulated Genes:")
print(upregulated[['GeneID', 'log2(FC)', 'P-adj']])

# Create the volcano plot
plt.figure(figsize=(10, 6))

# Plot downregulated genes
plt.scatter(downregulated['log2(FC)'], -np.log10(downregulated['P-adj']), color='blue', label='Downregulated', alpha=0.5)

# Plot upregulated genes
plt.scatter(upregulated['log2(FC)'], -np.log10(upregulated['P-adj']), color='red', label='Upregulated', alpha=0.5)

# Add labels and title
plt.xlabel('log2 Fold Change')
plt.ylabel('-log10(Adjusted P-value)')
plt.title('Volcano Plot')

# Add significance threshold lines
plt.axvline(x=2, color='black', linestyle='--')
plt.axvline(x=-2, color='black', linestyle='--')
plt.axhline(y=-np.log10(0.05), color='black', linestyle='--')

# Add legend
plt.legend()

# Show plot
plt.show()


import pandas as pd

file_path = "C:/Users/jisha/Downloads/Documents/Dengue_Disease_sate.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet1')

print(data.head())


# Print the column names to check if 'P-value' exists
print("Columns in the DataFrame:", data.columns)

# Check if 'pvalue' column exists and then proceed
if 'pvalue' in data.columns:
    data['-log10(P-value)'] = -np.log10(data['P-value'])
    print(data.head())
else:
    print("Column 'P-value' does not exist in the DataFrame")


# Add a column for -log10(pvalue)
data['-log10(P-adj)'] = -np.log10(data['P-adj'])

# Filter for significant genes with log2(FC) > 2.0 and p-adj < 0.05
data['significant'] = (data['log2(FC)'] > 2.0) & (data['P-adj'] < 0.05)
data['significant'] = (data['log2(FC)'] > 2.0) & (data['P-adj'] < 0.05)

# Create a column for significance in the original data
data['significant'] = ((data['log2(FC)'] > 2) | (data['log2(FC)'] < -2)) & (data['P-adj'] < 0.05)

# Filter the data based on downregulation and upregulation
downregulated = data[data['log2(FC)'] < -2]
upregulated = data[data['log2(FC)'] > 2]

# Print downregulated genes with p-adj value
print("Downregulated Genes:")
print(downregulated[['GeneID', 'log2(FC)', 'P-adj']])

# Print upregulated genes with p-adj value
print("\nUpregulated Genes:")
print(upregulated[['GeneID', 'log2(FC)', 'P-adj']])

# Create the volcano plot
plt.figure(figsize=(10, 6))

# Plot downregulated genes
plt.scatter(downregulated['log2(FC)'], -np.log10(downregulated['P-adj']), color='blue', label='Downregulated', alpha=0.5)

# Plot upregulated genes
plt.scatter(upregulated['log2(FC)'], -np.log10(upregulated['P-adj']), color='red', label='Upregulated', alpha=0.5)

# Add labels and title
plt.xlabel('log2 Fold Change')
plt.ylabel('-log10(Adjusted P-value)')
plt.title('Volcano Plot')

# Add significance threshold lines
plt.axvline(x=2, color='black', linestyle='--')
plt.axvline(x=-2, color='black', linestyle='--')
plt.axhline(y=-np.log10(0.05), color='black', linestyle='--')

# Add legend
plt.legend()

# Show plot
plt.show()


# Overlay Volcano Plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path_1 = "C:/Users/jisha/Downloads/Documents/2_months.xlsx"
file_path_2 = "C:/Users/jisha/Downloads/Documents/Dengue_Disease_sate.xlsx"
data1 = pd.read_excel(file_path_1, sheet_name='Sheet1')
data2 = pd.read_excel(file_path_2, sheet_name='Sheet1')

# Filter significant genes
significant_data1 = data1[(data1['log2(FC)'].abs() > 2) & (data1['P-adj'] < 0.05)]
significant_data2 = data2[(data2['log2(FC)'].abs() > 2) & (data2['P-adj'] < 0.05)]

# Create the overlay volcano plot
plt.figure(figsize=(10, 6))

# Plot significant genes from dataset 1
plt.scatter(significant_data1['log2(FC)'], -np.log10(significant_data1['P-adj']), color='blue', label='Dataset 1', alpha=0.5)

# Plot significant genes from dataset 2
plt.scatter(significant_data2['log2(FC)'], -np.log10(significant_data2['P-adj']), color='red', label='Dataset 2', alpha=0.5)

# Add labels and title
plt.xlabel('log2 Fold Change')
plt.ylabel('-log10(Adjusted P-value)')
plt.title('Overlay Volcano Plot')

# Add significance threshold lines
plt.axvline(x=2, color='black', linestyle='--')
plt.axvline(x=-2, color='black', linestyle='--')
plt.axhline(y=-np.log10(0.05), color='black', linestyle='--')

# Add legend
plt.legend()

# Show plot
plt.show()


# venn diagram

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Load the first dataset
file_path_1 = "C:/Users/jisha/Downloads/Documents/2_months.xlsx"
data1 = pd.read_excel(file_path_1, sheet_name='Sheet1')

# Load the second dataset
file_path_2 = "C:/Users/jisha/Downloads/Documents/Dengue_Disease_sate.xlsx"
data2 = pd.read_excel(file_path_2, sheet_name='Sheet1')

# Add a column for -log10(p-adj) if necessary
if 'P-adj' in data1.columns:
    data1['-log10(P-adj)'] = -np.log10(data1['P-adj'])

if 'P-adj' in data2.columns:
    data2['-log10(P-adj)'] = -np.log10(data2['P-adj'])

# Create a column for significance
data1['significant'] = ((data1['log2(FC)'] > 2) | (data1['log2(FC)'] < -2)) & (data1['P-adj'] < 0.05)
data2['significant'] = ((data2['log2(FC)'] > 2) | (data2['log2(FC)'] < -2)) & (data2['P-adj'] < 0.05)

# Extract lists of significant genes
significant_genes1 = set(data1[data1['significant']]['GeneID'])
significant_genes2 = set(data2[data2['significant']]['GeneID'])

# Create the Venn diagram
plt.figure(figsize=(8, 8))
venn2([significant_genes1, significant_genes2], ('2 Months', 'Dengue Disease State'))

# Add title
plt.title('Venn Diagram of Significant Genes')

# Show plot
plt.show()


# Heatmap plot

import seaborn as sns

# Select significant genes and create a dataframe
significant_genes = list(significant_genes1.union(significant_genes2))
heatmap_data = data1[data1['GeneID'].isin(significant_genes)].set_index('GeneID')[['log2(FC)']].join(
    data2[data2['GeneID'].isin(significant_genes)].set_index('GeneID')[['log2(FC)']], lsuffix='_1', rsuffix='_2'
).dropna()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
plt.title('Heatmap of Significant Genes')
plt.xlabel('Condition')
plt.ylabel('Gene')
plt.show()


# Scatter plot

# Merge datasets on GeneID
merged_data = data1.merge(data2, on='GeneID', suffixes=('_1', '_2'))

# Scatter plot

plt.figure(figsize=(10, 6))
plt.scatter(merged_data['log2(FC)_1'], merged_data['log2(FC)_2'], alpha=0.5)
plt.xlabel('log2 Fold Change - Dataset 1')
plt.ylabel('log2 Fold Change - Dataset 2')
plt.title('Scatter Plot of log2 Fold Changes')
plt.show()


# Bar plot

# Count upregulated and downregulated genes
upregulated_count_1 = sum(data1['log2(FC)'] > 2)
downregulated_count_1 = sum(data1['log2(FC)'] < -2)
upregulated_count_2 = sum(data2['log2(FC)'] > 2)
downregulated_count_2 = sum(data2['log2(FC)'] < -2)

# Bar plot
labels = ['Upregulated - Dataset 1', 'Downregulated - Dataset 1', 'Upregulated - Dataset 2', 'Downregulated - Dataset 2']
counts = [upregulated_count_1, downregulated_count_1, upregulated_count_2, downregulated_count_2]

plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color=['red', 'blue', 'red', 'blue'])
plt.ylabel('Number of Genes')
plt.title('Number of Upregulated and Downregulated Genes')
plt.xticks(rotation=45)
plt.show()


# Principal Component Analysis

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Example data
data1 = pd.DataFrame({
    'GeneID': ['gene1', 'gene2', 'gene3', 'gene4'],
    'log2(FC)': [2.5, -3.0, 1.2, -4.5],
    'P-adj': [0.01, 0.04, 0.05, 0.001]
})
data2 = pd.DataFrame({
    'GeneID': ['gene1', 'gene2', 'gene5', 'gene6'],
    'log2(FC)': [2.8, -2.9, 1.5, -3.8],
    'P-adj': [0.02, 0.03, 0.04, 0.002]
})

# Combine datasets for PCA
combined_data = pd.concat([data1.set_index('GeneID')['log2(FC)'], data2.set_index('GeneID')['log2(FC)']], axis=1, keys=['Dataset1', 'Dataset2']).fillna(0)

# Verify combined data
print(combined_data)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(combined_data)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot PCA
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Gene Expression')
plt.show()


# Density plot

# Density plots for log2 fold changes
plt.figure(figsize=(10, 6))
sns.kdeplot(data1['log2(FC)'], fill=True, label='Dataset 1', color='blue')
sns.kdeplot(data2['log2(FC)'], fill=True, label='Dataset 2', color='red')
plt.xlabel('log2 Fold Change')
plt.ylabel('Density')
plt.title('Density Plot of log2 Fold Changes')
plt.legend()
plt.show()


# Correlation plot

plt.figure(figsize=(10, 6))
sns.jointplot(x='log2(FC)_1', y='log2(FC)_2', data=merged_data, kind='reg', color='purple')
plt.xlabel('log2 Fold Change - Dataset 1')
plt.ylabel('log2 Fold Change - Dataset 2')
plt.title('Correlation Plot of log2 Fold Changes')
plt.show()


# Box plot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming data1 and data2 are your datasets loaded as pandas DataFrames

# Add a column to differentiate the datasets
data1['Dataset'] = 'Dataset 1'
data2['Dataset'] = 'Dataset 2'

# Combine the two datasets for plotting
combined_data = pd.concat([data1[['log2(FC)', 'Dataset']], data2[['log2(FC)', 'Dataset']]])

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Dataset', y='log2(FC)', data=combined_data, palette=['blue', 'red'])
plt.xlabel('Condition')
plt.ylabel('log2 Fold Change')
plt.title('Box Plot of log2 Fold Changes')
plt.show()


# MA plot

import matplotlib.pyplot as plt

# Calculate mean expression and log2 fold change
data1['mean_expression'] = data1[['log2(FC)']].mean(axis=1)
data2['mean_expression'] = data2[['log2(FC)']].mean(axis=1)

plt.figure(figsize=(10, 6))

# Plot MA plot
plt.scatter(data1['mean_expression'], data1['log2(FC)'], alpha=0.5, label='Dataset 1', color='blue')
plt.scatter(data2['mean_expression'], data2['log2(FC)'], alpha=0.5, label='Dataset 2', color='red')

plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Mean Expression')
plt.ylabel('log2 Fold Change')
plt.title('MA Plot')
plt.legend()
plt.show()

# Statistical tests
# Ttest

from scipy.stats import ttest_ind

# Perform t-test
t_stat, p_val = ttest_ind(data1['log2(FC)'], data2['log2(FC)'])
print(f"T-test: t-statistic = {t_stat}, p-value = {p_val}")


# ANOVA

from scipy.stats import f_oneway

# Assuming 'log2(FC)' is the column representing gene expression levels and 'Dataset' is the categorical variable indicating the dataset
dataset1_expression = data1['log2(FC)']
dataset2_expression = data2['log2(FC)']

# Perform ANOVA test
f_statistic, p_value = f_oneway(dataset1_expression, dataset2_expression)

print("ANOVA Results:")
print("F-statistic:", f_statistic)
print("p-value:", p_value)

if p_value < 0.05:
    print("There are significant differences in gene expression levels between the two datasets.")
else:
    print("There are no significant differences in gene expression levels between the two datasets.")


# GSEA

import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt

# Load your data
file_path_1 = "C:/Users/jisha/Downloads/Documents/2_months.xlsx"
file_path_2 = "C:/Users/jisha/Downloads/Documents/Dengue_Disease_sate.xlsx"
data1 = pd.read_excel(file_path_1, sheet_name='Sheet1')
data2 = pd.read_excel(file_path_2, sheet_name='Sheet1')

# Merge datasets based on GeneID for consistency
merged_data = data1.merge(data2, on='GeneID', suffixes=('_1', '_2'))

# Prepare the expression data
# Assuming 'log2(FC)' as the expression values
data_for_gsea = merged_data[['GeneID', 'log2(FC)_1', 'log2(FC)_2']].dropna()
data_for_gsea.set_index('GeneID', inplace=True)

# Path to the GMT file
gmt_file_path = 'C:/Users/jisha/Downloads/h.all.v2023.2.Hs.symbols.gmt'

# Rank the genes based on log2(FC) values
ranked_genes_1 = data_for_gsea['log2(FC)_1'].sort_values(ascending=False)
ranked_genes_2 = data_for_gsea['log2(FC)_2'].sort_values(ascending=False)

# Perform GSEA analysis for each condition
results_1 = gp.prerank(
    rnk=ranked_genes_1,
    gene_sets=gmt_file_path,
    outdir='gsea_results_2_months',
    min_size=15,
    max_size=500,
    permutation_num=100,  # Number of permutations
    verbose=True  # Consider setting it to False for less output
)

results_2 = gp.prerank(
    rnk=ranked_genes_2,
    gene_sets=gmt_file_path,
    outdir='gsea_results_dengue',
    min_size=15,
    max_size=500,
    permutation_num=100,  # Number of permutations
    verbose=True  # Consider setting it to False for less output
)

# Print results
print("GSEA results for 2 months dataset:\n", results_1.res2d)
print("GSEA results for Dengue Disease State dataset:\n", results_2.res2d)

# Inspect the structure of the results
print("Keys in results_1.results:\n", results_1.results.keys())
print("Keys in results_2.results:\n", results_2.results.keys())

# Function to plot GSEA results
def plot_gsea(results, ranked_genes, condition_name):
    top_gene_set = results.res2d.index[0]
    if top_gene_set not in results.results:
        print(f"Top gene set '{top_gene_set}' not found in results.")
        return
    
    # Ensure that the key names match the structure
    hits = results.results[top_gene_set]['hits']
    nes = results.res2d.loc[top_gene_set, 'nes']
    pval = results.res2d.loc[top_gene_set, 'pval']
    fdr = results.res2d.loc[top_gene_set, 'fdr']
    RES = results.results[top_gene_set]['res']

    # Use the gseaplot function correctly
    gp.plot.gseaplot(rank_metric=ranked_genes,
                     term=top_gene_set,
                     hits=hits,
                     nes=nes,
                     pval=pval,
                     fdr=fdr,
                     RES=RES,
                     ofname=f"{condition_name}_top_enriched_plot.png")
    plt.show()

# Plot the top enriched gene sets for both conditions
plot_gsea(results_1, ranked_genes_1, '2_months')
plot_gsea(results_2, ranked_genes_2, 'dengue_disease_state')

# Display the top results for each condition
print("GSEA results for 2 months dataset:\n", results_1.res2d.head())
print("GSEA results for Dengue Disease State dataset:\n", results_2.res2d.head())
# Display the plot
results_1.res2d.plot(kind='bar')
plt.show()  # Ensure this command is present to display the plot
