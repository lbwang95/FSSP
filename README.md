# FSSP

Efficiently Answering Semantic Optimal Sequenced Route Queries with Dynamic Candidates

### File Description

There are three folders: code/, data/, model/. The folder ./model stores the paraphrase-multilingual-mpnet-base-v2 that creates embedding vectors. The folder ./code includes the code for the three algorithms, DROMC, ROSE, and FSSP. All the C++ implementations first call a Python program for the model and obtain embedding vectors from the Python program. The folder ./data stores the necessary graphs and queries for each network: NYC, BAY, COL, FLA, and exa (for the running example used in the paper). 

### Execution

There are four algorithms, BSM, LSD, SDALT, and Dijkstra, as stated in Section 6. All algorithms use the following command.

./[algorithm] [network] [query file] [regular expression]

If the third parameter [regular expression] does not exist, the program will read it from each line of each query, which means that the regular language of each query is different. If it exists, the regular language for each query in this query file is all set to the third parameter [regular expression]. An example command is "./BSM exa q1 \\(A1\\|A2\\)*A4\\(A2*\\|A3*\\)", which runs the example in our paper. Note that we need to use the escape character "\\" before |, (, ). The network includes NYC, BAY, COL, FLA, and exa with corresponding data in the folder data/. The last one represents the example used in our paper. The query files are stored under each network folder "data/[network]/". The regular expression uses *, |, (, ) following the standard, and the alphabet $\Sigma=\{A1, A2, A3, A4, A5, B1, B2, B3, B4, B5, C1, C2, C3, C4, C5, D1, D2, D3, D4, D5\}$. For the example in our paper, $A1, A2, A3, A4$ stand for $\alpha,\beta,\theta,\gamma$, respectively. We can run the default experiments by the following command (detailed in our paper).

./BSM NYC q1 \\(D1\\|C1\\|B1\\|B5\\|A5\\|A1\\|D5\\|C5\\|D3\\|A3\\)*

During the execution, the screen will show some auxiliary information, including the time cost for each algorithm in microseconds and the index size in MB. The query answers will be stored in the folder "data/[network]/" with the suffix "-Results" when the program finishes.

### Parameters

To use different networks, we vary the [network]. For example, ./BSM NYC q10 \\(D1\\|C1\\|B1\\|B5\\|A5\\|A1\\|D5\\|C5\\|D3\\|A3\\)*, ./BSM BAY q10 \\(D1\\|C1\\|B1\\|B5\\|A5\\|A1\\|D5\\|C5\\|D3\\|A3\\)*, ...

To test the effect of distances, we simply vary the [query file] from q1 to q10 since they contain queries with increasing distances: ./BSM NYC q1 \\(D1\\|C1\\|B1\\|B5\\|A5\\|A1\\|D5\\|C5\\|D3\\|A3\\)*, ./BSM NYC q2 \\(D1\\|C1\\|B1\\|B5\\|A5\\|A1\\|D5\\|C5\\|D3\\|A3\\)*, ...

To test the effect of regular languages, we vary the number of allowed labels in Exp-2: ./BSM NYC q10 \\(D1\\)*, ./BSM NYC q10 \\(D1\\|C1\\)*, ./BSM NYC q10 \\(D1\\|C1\\|B1\\)*, ...

To test the effect of DFA states, we vary the regular languages as follows: ./BSM NYC q10 D1*, ./BSM NYC q10 D1*C1*, ...

Note that the sorted labels from frequent to less frequent ones are D1, C1, B1, B5, A5, A1, D5, C5, D3, A3, C3, B3, D4, D2, C2, A2, B2, B4, A4, C4.

### Datasets

All the data used in the experiments are stored in the folder "data/". Under it, there is a folder for each network. 

In each folder, the file "USA-road-l.[network name].gr" contains the road network data. The file "USA-road-d.[network name].co" includes all the coordinates of vertices. Their formats follow the description in DIMACS (http://www.dis.uniroma1.it/~challenge9). The file "order.txt" is used for the LSD index. The files "q1"-"q10" are the query datasets with increasing query distances, as introduced in Section 6. Each line specifies the source vertex id and destination vertex id of a query. 

The file "landmarks.txt" includes the number of landmarks in the first line and the vertex id of each landmark in the following lines. The "landmarksDis.txt" contains the unconstrained distances from each vertex to each landmark. Each line includes the distances from a vertex (whose id corresponds to the line number) to all landmarks in the order of their appearances in "landmarks.txt". Since Github has file size limit, we split the "landmarksDis.txt" for COL and FLA, which can easily restored.
