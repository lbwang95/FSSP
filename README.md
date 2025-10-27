# FSSP

Efficiently Answering Semantic Optimal Sequenced Route Queries with Dynamic Candidates

### File Structure

There are three folders: code/, data/, model/. The folder ./model stores the paraphrase-multilingual-mpnet-base-v2 that creates embedding vectors, which can be downloaded from https://drive.google.com/file/d/1YV9DX5loU9guqHMEPU_ZtDT-SXSNIHnc/view?usp=sharing (See the README in ./model). The folder ./code includes the code for the three algorithms, DROMC, ROSE, and FSSP. All the C++ implementations first call a Python program for the model and obtain embedding vectors from the Python program. The folder ./data stores the necessary graphs and queries for each network: NYC, BAY, COL, FLA, and exa (for the running example used in the paper). 

### Execution

As stated in experiments, there are three algorithms: DROMC, ROSE, and FSSP.

./[algorithm] [network] [requirement file] [tolerance parameter \alpha]

[network] can be NYC, BAY, COL, FLA, and exa. [requirement file] can be req1 - req5 or specified by users, and it should be put under data/[network]/. The default setting uses the following command:

./FSSP NYC req3 0.05

The default command uses query file q5, which includes 1,000 source-target pairs. The requirement file includes k lines corresponding to k requirements expressed in natural languages. For the generation details of the data, please refer to the experiments section.

### Datasets

All the data used in the experiments are stored in the folder "data/". Under it, there is a folder for each network. 

In each folder, the file "USA-road-d.[network name].gr" contains the road network graph data. The file "USA-road-d.[network name].co" includes all the coordinates of vertices. Their formats follow the description in DIMACS (http://www.dis.uniroma1.it/~challenge9). The file "order.txt" is used for the H2H index. The files "q1"-"q5" are the query datasets with increasing query distances. Each line specifies the source vertex id and destination vertex id of a query. 

The files "LLL", "LML", ... "HHH" are also requirement files, each of which includes k lines corresponding to k requirements expressed in natural languages. "category_embeddings.txt“ includes vectors of 768 dimensions for all |C| categories. "POInames_ID.txt" and "Categories_ID.txt" include POI names and category names with IDs. "poi_node_mapping.txt" includes POI information for each vertex. Specifically, several lines form a group for each vertex, which first contain [vertex ID]-, then [POI1 ID]<[category1 ID]^[category2 ID]..., then [POI2_ID]<[]...
