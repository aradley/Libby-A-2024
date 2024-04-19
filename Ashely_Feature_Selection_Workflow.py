
##### Dependencies #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import pairwise_distances

import cESFW

import plotly.express as px 
Colours = px.colors.qualitative.Dark24
Colours.remove('#222A2A')
Colours = np.concatenate((Colours,Colours))

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore
import seaborn as sns

# Adapted ES feature weighting function

def Feature_Weighting(ESS_Threshold, EPs_Threshold, Min_Edges, ESSs, EPs, Used_Features):
    Absolute_ESSs = np.absolute(ESSs)
    ## Network filtering
    # Initial network filtering will be done by pruning the positive EP and ESSs threshold graph until
    # all nodes of the graph have a minimum number of edges. The "OR" logic for thresholding is used
    # here because we want to exclude edges that have negative EPs or low ESSs. Essentially we are looking for a sub-graph
    # of the entire gene network where genes are connected if they show significant (EP > 0) enrichment with one another, while
    # adding heuristic ESS threshold to filter very low positive enrichments (Absolute_ESSs < ESS_Threshold).
    Mask_Inds = np.where((EPs <= EPs_Threshold) | (Absolute_ESSs < ESS_Threshold))
    ESSs_Graph = Absolute_ESSs.copy()
    ESSs_Graph[Mask_Inds] = 0
    EPs_Graph = EPs.copy()
    EPs_Graph[Mask_Inds] = 0
    ## Prune poorly connected genes
    Keep_Features = np.array([])
    while Keep_Features.shape[0] < EPs_Graph.shape[0]:
        print("Genes remaining: " + str(EPs_Graph.shape[0]))
        Keep_Features = np.where(np.sum(EPs_Graph > 0,axis=0) > Min_Edges)[0]
        Used_Features = Used_Features[Keep_Features]
        #
        Absolute_ESSs = Absolute_ESSs[np.ix_(Keep_Features,Keep_Features)]
        EPs = EPs[np.ix_(Keep_Features,Keep_Features)]
        Mask_Inds = np.where((EPs <= EPs_Threshold) | (Absolute_ESSs < ESS_Threshold))
        ESSs_Graph = Absolute_ESSs.copy()
        ESSs_Graph[Mask_Inds] = 0
        EPs_Graph = EPs.copy()
        EPs_Graph[Mask_Inds] = 0
        Keep_Features = np.where(np.sum(EPs_Graph > 0,axis=0) > Min_Edges)[0]
    ## Now we will get the weighted node centrality of each node in the graph. For this, we will use "AND" logic instead of "OR".
    # We do this because although the remaining genes meet the criteria of forming a connected graph with a minimum number of edges, some
    # gene/nodes may still have more non-significant edges than significant ones (EPs < 0) when considering all possible gene interactions,
    # meaning that the presence of these genes/nodes introduce more random noise than they contribute structure, and hence their importance 
    # weights should be penalised by the negative EPs. As previously, we add a heuristic ESS threshold 
    # to filter very low positive enrichments (Absolute_ESSs < ESS_Threshold).
    Mask_Inds = np.where((EPs <= EPs_Threshold) & (Absolute_ESSs < ESS_Threshold))
    Masked_ESSs = Absolute_ESSs.copy()
    Masked_ESSs[Mask_Inds] = 0
    Masked_EPs = EPs.copy()
    Masked_EPs[Mask_Inds] = 0
    ## Feature weighting via weighted node centrality
    Feature_Weights = np.average(Absolute_ESSs,weights=Masked_EPs,axis=0)
    Significant_Genes_Per_Gene = (Masked_EPs > 0).sum(1)
    Normalised_Network_Feature_Weights = Feature_Weights/Significant_Genes_Per_Gene
    return Used_Features, ESSs, EPs, ESSs_Graph, EPs_Graph, Masked_ESSs, Masked_EPs, Feature_Weights, Normalised_Network_Feature_Weights, Significant_Genes_Per_Gene

##### Dependencies #####


##### Data pre-processing #####

Ashely_path = "/Users/radleya/Dropbox (The Francis Crick)/BriscoeJ/Radleya/Asheley_Data/Data_2/"

Ashely_Data = pd.read_csv(Ashely_path+"Ashley_Subset.csv",header=0,index_col=0)

Keep_Genes = Ashely_Data.columns[np.where(np.sum(Ashely_Data > 0,axis=0) > 3)[0]]
Keep_Genes.shape[0]

Ashely_Data = Ashely_Data[Keep_Genes]

### Feature normalisation ###

## Prior to using cESFW, data must be scaled/normalised such that every feature only has values between 0 and 1.
# How this is done is ultimitely up to the user. However, for scRNA-seq data, we tend to find that the following relitively simple
# normalisation approach yeilds good results.

## Note, cESFW takes each row to be a sample and each column to be a feature. Hence, in this example, each row of Human_Embryo_Counts
# is a cell and each colum is gene.

## Create the scaled matrix from the scRNA-seq counts matrix
Scaled_Matrix = np.log2(Ashely_Data.copy()+1)

## Optional: Log transform the data. Emperically we find that in most cases, log transformation of the data
# appears to lead to poorer results further downstream in most cases. However, in some datasets
# we have worked with, it has lead to improved results. This is obviously dependent on what downstream analysis
# the user chooses to do and how they do it, but we recommend starting without any log transformation (hence the
# next line of code being commented out).
#### Scaled_Matrix = np.log2(Scaled_Matrix+1) ####

## Clip the top 2.5 percent of observed values for each gene to mitigate the effect of unusually high
# counts observations.
Upper = np.percentile(Scaled_Matrix,97.5,axis=0)
Upper[np.where(Upper == 0)[0]] = np.max(Scaled_Matrix,axis=0)[np.where(Upper == 0)[0]]
Scaled_Matrix = Scaled_Matrix.clip(upper=Upper,axis=1)

## Normalise each feature/gene of the clipped matrix.
Normalisation_Values = np.max(Scaled_Matrix,axis=0)
Scaled_Matrix = Scaled_Matrix / Normalisation_Values

### Run cESFW ###

## Given the scaled matrix, cESFW will use the following function to extract all the non-zero values into a single vector. We do this
# because ES calculations can completely ignore 0 values in the data. For sparse data like scRNA-seq data, this dramatically reduces the memory
# required, and the number of calculations that need to be carried out. For relitively dense data, this step will still need to be carried
# out to use cESFW, but will provide little benifit computationally.

## path: A string path pre-designated folder to deposit the computationally efficient objects. E.g. "/mnt/c/Users/arthu/Test_Folder/"
## Scaled_Matrix: The high dimensional DataFrame whose features have been scaled to values between 0 and 1. Format must be a Pandas DataFrame.
## Min_Minority_State_Cardinality: The minimum value of the total minority state mass that a feature contains before it will be automatically
# removed from the data, and hence analysis.

cESFW.Create_ESFW_Objects(Ashely_path, Scaled_Matrix, Min_Minority_State_Cardinality = 3)

## Now that we have the compute efficient object, we can calculate the ESSs and EPs matricies. The ESSs matrix provides the pairwise 
# Entropy Sort Scores for each gene in the data. THe EPs matrix provides the EPs pairwise for each gene.

ESSs, EPs = cESFW.Parallel_Calculate_ESS_EPs(Ashely_path)
np.save(Ashely_path + "ESSs.npy",ESSs)
np.save(Ashely_path + "EPs.npy",EPs)

#### Perform feature weighting with ESS and EP objects

Used_Features = np.load(Ashely_path + "Used_Features.npy",allow_pickle=True)
ESSs = np.load(Ashely_path + "ESSs.npy")
EPs = np.load(Ashely_path + "EPs.npy")

ESS_Threshold = 0.02
EPs_Threshold = 0
Min_Edges = 5

Subset_Used_Features, ESSs, EPs, ESSs_Graph, EPs_Graph, Masked_ESSs, Masked_EPs, Feature_Weights, Normalised_Network_Feature_Weights, Significant_Genes_Per_Gene = Feature_Weighting(ESS_Threshold, EPs_Threshold, Min_Edges, ESSs, EPs, Used_Features)

# Some genes that we know are important in early human embryo development

Known_Important_Genes = np.array(["SHH","TBXT","SOX10","SOX9","LMO4","HOXA7","MSX1","FABP5"])
Known_Important_Genes = np.unique(Known_Important_Genes)
Known_Important_Genes[np.isin(Known_Important_Genes,Subset_Used_Features) == 0]
Known_Important_Gene_Inds = np.where(np.isin(Subset_Used_Features,Known_Important_Genes))[0]

## Running the next two lines shows us that genes that we know to be important for early human development, are amongst the highest
# ranked genes when considering the normalised network feature weights.
np.where(np.isin(np.argsort(-Feature_Weights),Known_Important_Gene_Inds))[0]
np.where(np.isin(np.argsort(-Normalised_Network_Feature_Weights),Known_Important_Gene_Inds))[0]

## Take the top 500 ranked genes.
Use_Inds = np.argsort(-Normalised_Network_Feature_Weights)[np.arange(500)] 
Selected_Genes = Subset_Used_Features[Use_Inds]
Selected_Genes.shape[0]

## Sometimes when picking how many of the top ranked genes to take, it is useful to see if important known
# markers are captured by your threshold.
np.isin(Known_Important_Genes,Selected_Genes)

## Visualise the gene clusters on a UMAP.
Neighbours = 30
Dist = 0.1
Gene_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Masked_ESSs[np.ix_(Use_Inds,Use_Inds)])

Plot_Important_Inds = np.where(np.isin(Selected_Genes,Known_Important_Genes))[0]
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.title("Colour = Feature_Weights", fontsize=20)
plt.scatter(Gene_Embedding[:,0],Gene_Embedding[:,1],s=7,c=Feature_Weights[Use_Inds],vmax=0.1)
plt.colorbar()
plt.scatter(Gene_Embedding[Plot_Important_Inds,0],Gene_Embedding[Plot_Important_Inds,1],s=20,c="red")
plt.xlabel("UMAP 1",fontsize=16)
plt.ylabel("UMAP 2",fontsize=16)
plt.subplot(1,2,2)
plt.title("Colour = Normalised_Network_Feature_Weights", fontsize=20)
plt.scatter(Gene_Embedding[:,0],Gene_Embedding[:,1],s=7,c=Normalised_Network_Feature_Weights[Use_Inds])#,vmax=0.0002)
plt.colorbar()
plt.scatter(Gene_Embedding[Plot_Important_Inds,0],Gene_Embedding[Plot_Important_Inds,1],s=20,c="red")
# plt.scatter(Gene_Embedding[:,0],Gene_Embedding[:,1],s=7,c=np.log2(Feature_Weights*Significant_Genes_Per_Gene)[Use_Inds])
plt.xlabel("UMAP 1",fontsize=16)
plt.ylabel("UMAP 2",fontsize=16)
# plt.savefig(human_embryo_path + "Gene_Cluster.png",dpi=600)
# plt.close()
plt.show()

## We will now subset the original scRNA-seq data down to a different clusters of genes to identify embeddings that are consistent
# with experimentally determined knowledge.

Keep_Genes = np.where((Gene_Embedding[:,0] > 5))[0]
Cluster_Use_Gene_IDs = Selected_Genes[Keep_Genes][np.argsort(-Feature_Weights[Use_Inds][Keep_Genes])]

Reduced_Input_Data = np.log2(Ashely_Data[Cluster_Use_Gene_IDs]+1)
Reduced_Input_Data.shape[1]

Neighbours = 20
Dist = 0.1

Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
Embedding = Embedding_Model.embedding_

plt.scatter(Embedding[:,0],Embedding[:,1],s=7)
plt.show()

### Save the selected top genes and embedding for future plotting
# np.save(Ashely_path+"Embedding2.npy",Embedding)
# np.save(Ashely_path+"Saved_cESFW_Genes2.npy",Cluster_Use_Gene_IDs)

### Plotting ###

Embedding = np.load(Ashely_path+"Embedding2.npy")
Cluster_Use_Gene_IDs = np.load(Ashely_path+"Saved_cESFW_Genes2.npy",allow_pickle=True)

# Plot some gene of interest
Plot_Genes = np.array(["NKX6-1","NKX2-2","NKX2-6","NKX6-2","OLIG2","SFRP1","PAX7","SOX9","PAX6","SHH","FOXA2","FABP5","SOX1","CA2","NKX1-2","MLLT3","MSGN1","TBXT","FGF8","CLDN1"])

Plot_Genes = Plot_Genes[np.isin(Plot_Genes,Ashely_Data.columns)]
for i in np.arange(Plot_Genes.shape[0]):
    plt.figure(figsize=(8,8))
    Gene = Plot_Genes[i]
    plt.title(Gene,fontsize=18)
    plt.scatter(Embedding[:,0],Embedding[:,1],s=7,c=np.log2(Ashely_Data[Gene]+1),cmap="seismic")
    # plt.savefig(Ashely_path + "Plots/" + Gene + ".png",dpi=900)
    # plt.close()

plt.show()

# Cluster samples according to the top ranked genes identified by ES
distmat = pairwise_distances(np.log2(Ashely_Data[Cluster_Use_Gene_IDs]+1),metric='correlation')
pca = PCA(n_components=30)
pca.fit(distmat)
Use_Comps = pca.components_.T[:,0:10]

kmeans = KMeans(n_clusters=15, random_state=0, n_init="auto").fit(Use_Comps)
Cluster_Labels = kmeans.labels_
Cluster_Labels = Cluster_Labels.astype("str")

Unique_Cluster_Labels = np.unique(Cluster_Labels)

plt.figure(figsize=(6,6))
for i in np.arange(Unique_Cluster_Labels.shape[0]):
    IDs = np.where(Cluster_Labels == Unique_Cluster_Labels[i])
    plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=5,c=Colours[i],label=Unique_Cluster_Labels[i])
#
 
plt.legend()
plt.show()

## Reannotate clusters according to gene signatures and known biological domains.
Consolidated_Clusters = Cluster_Labels.copy()
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["9","13","5","3","12"])))] = "Mesoderm"
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["2","6","1"])))] = "PS"
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["0"])))] = "NC"
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["4"])))] = "NT dorsal"
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["7"])))] = "NT ventral"
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["8"])))] = "Pre-NT 1"
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["10"])))] = "FP"
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["11"])))] = "Pre-NT 2"
Consolidated_Clusters[np.where(np.isin(Consolidated_Clusters,np.array(["14"])))] = "NT medial"

Unique_Consolidated_Clusters = np.unique(Consolidated_Clusters)
Row_Colours = Consolidated_Clusters.copy()

plt.figure(figsize=(6,6))
for i in np.arange(Unique_Consolidated_Clusters.shape[0]):
    IDs = np.where(Consolidated_Clusters == Unique_Consolidated_Clusters[i])
    plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=5,c=Colours[i],label=Unique_Consolidated_Clusters[i])
    Row_Colours[IDs] = Colours[i]
#

plt.legend(ncol=2)
plt.show()

## Create pseudo bulk data by taking the means gene expression values of samples in each cluster.
Mean_Expression = pd.DataFrame(np.zeros((Unique_Consolidated_Clusters.shape[0],Ashely_Data.shape[1])),index=Unique_Consolidated_Clusters,columns=Ashely_Data.columns)
for i in np.arange(Unique_Consolidated_Clusters.shape[0]):
    IDs = np.where(Consolidated_Clusters == Unique_Consolidated_Clusters[i])
    Mean_Expression.iloc[i] = Ashely_Data.iloc[IDs].mean(axis=0)


## Create and ranked gene list of genes in each cluster by calculating the Entropy Sort Score (ESS) for genes into each cluster
Used_Features = np.load(Ashely_path + "Used_Features.npy",allow_pickle=True)
Ranked_Genes_Table = pd.DataFrame(np.zeros((Used_Features.shape[0],len(Unique_Consolidated_Clusters))),index=np.arange(Used_Features.shape[0]),columns=Unique_Consolidated_Clusters)

Top_Genes = []
Top_Number = 20
for i in np.arange(len(Unique_Consolidated_Clusters)):
    Annotation_Labels = np.zeros(Ashely_Data.shape[0])
    Annotation_Labels[np.where(Consolidated_Clusters == Unique_Consolidated_Clusters[i])[0]] = 1
    Individual_ESSs, Individual_EPs = cESFW.Calculate_Individual_ESS_EPs(Annotation_Labels,Ashely_path)
    Ranked_Genes = Used_Features[np.argsort(-Individual_ESSs)] 
    Top_Genes.append(Ranked_Genes[np.arange(Top_Number)])
    Ranked_Genes_Table[Unique_Consolidated_Clusters[i]] = Ranked_Genes
#
# Save ranked gene lists for each cluster
Ranked_Genes_Table.to_csv(Ashely_path + "Paper_Stuff/" + "ES_Cluster_Ranked_Genes_Lists.csv")    

Top_Genes = np.concatenate(Top_Genes)
Top_Genes = np.unique(Top_Genes)
Top_Genes.shape[0]
## Add MLLT3 as it is the only gene from the perturb-seq screen not in the top 20 genes of any of our clusters. It is raked
# 38 in the Primitive Streak (PS) cluster.
Top_Genes = np.append(Top_Genes,"MLLT3")

## Save UMAP cluster plot
plt.figure(figsize=(6,6))
for i in np.arange(Unique_Consolidated_Clusters.shape[0]):
    IDs = np.where(Consolidated_Clusters == Unique_Consolidated_Clusters[i])
    plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=5,c=Colours[i],label=Unique_Consolidated_Clusters[i])
    Row_Colours[IDs] = Colours[i]
#  
plt.legend(ncol=2)
plt.savefig(Ashely_path + "Paper_Stuff/" + "ES_UMAP" + ".png",dpi=900)
plt.close()
plt.show()

## Save heatmap of z-score normalised pseudobulk samples and top ranked genes.
Mean_Row_Colours = Colours[np.arange(Mean_Expression.shape[0])]
sns.clustermap(Mean_Expression[Top_Genes].apply(zscore),cmap="seismic",row_colors=Mean_Row_Colours,xticklabels=True)#,metric="correlation")
plt.savefig(Ashely_path + "Paper_Stuff/" + "ES_Heatmap" + ".png",dpi=900)
plt.close()
plt.show()

# Save ES UMAP Cords
pd.DataFrame(Embedding,index=Ashely_Data.index,columns=["UMAP 1", "UMAP 2"]).to_csv(Ashely_path + "Paper_Stuff/" + "ES_UMAP_Cords.csv")
# Save cluster labels
pd.DataFrame(Consolidated_Clusters,index=Ashely_Data.index,columns=["Cluster IDs"]).to_csv(Ashely_path + "Paper_Stuff/" + "ES_Cluster_IDs.csv")
# Save z-score heatmap raw data
Mean_Expression[Top_Genes].apply(zscore).to_csv(Ashely_path + "Paper_Stuff/" + "ES_Heatmap_Data.csv")

