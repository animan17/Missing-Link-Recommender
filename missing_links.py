import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict

IGNORE_VALUE = -1

def create_graph_from_csv(file_path):
    """
    Create a directed graph from a CSV file.
    """
    df = pd.read_csv(file_path)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        source = row.iloc[0]
        targets = [target for target in row[1:] if pd.notna(target)]
        G.add_node(source)
        for target in targets:
            G.add_edge(source, target)
    return G

def create_adjacency_matrix(G):
    """
    Create an adjacency matrix from a directed graph.
    """
    nodes_to_remove = [u for u in G.nodes() if u == "nan" or len(list(G.neighbors(u))) == 0]

    for u in nodes_to_remove:
        G.remove_node(u)


    nodes = G.nodes()

    num_nodes = len(nodes)        
    adj_mat = np.full((num_nodes, num_nodes), IGNORE_VALUE)
    node_to_index = {node: i for i, node in enumerate(nodes)}


    for u in G.nodes():
        for v in G.neighbors(u):
            adj_mat[node_to_index[u]][node_to_index[v]] = 1
        adj_mat[node_to_index[u]][node_to_index[u]] = 0
    return adj_mat, node_to_index, num_nodes, list(enumerate(nodes))

def find_svd(mat):
    """
    Perform Singular Value Decomposition (SVD) on the matrix.
    """

    matzero = mat
    matzero[matzero == IGNORE_VALUE] = 0

    dim = matzero.shape

    print(np.sum(matzero))


    U, S, Vh = np.linalg.svd(matzero, full_matrices=True)

    od = OrderedDict()

    tot = 0

    row = 0
    for i in S:
        od[i]=row
        row+=1
        tot+=i*i

    latent = []

    sum = 0
    for key,val in od.items():
        sum += key*key
        if sum>0.9*tot:
            latent.append(val)

    U = np.delete(U, latent, axis=1)
    S = np.delete(S, latent, axis=0)
    Vh = np.delete(Vh, latent, axis=0)

    a0 = U
    b0 = np.dot(np.diag(S), Vh)


    return find_ab(mat, a0, b0, a0.shape[0], a0.shape[1])

def find_ab(mat, a, b, r, c):
    """
    Find matrices A and B using the given matrix and initialized matrices A and B.
    """

    epochs = 10
    alpha = 0.01

    for i in range(epochs):
        pred = np.dot(a, b)

        loss = np.zeros((r,r))

        l = 0

        for i in range(r):
            for j in range(r):
                if mat[i][j] == IGNORE_VALUE:
                    continue

                val = 0
                for k in range(c):
                    val += a[i][k] * b[k][j]

                loss[i][j] += mat[i][j] - val

                l = loss[i][j]**2

        
        # Compute gradients
        gradient_p = (2/r) * np.dot(loss, b.T)
        gradient_q = (2/r) * np.dot(a.T, loss)
        
        # Update p and q using the gradients
        a -= alpha * gradient_p
        b -= alpha * gradient_q
        
        if i % 100 == 0:  # Print loss every 100 iterations
            print(f"Iteration {i}: Loss = {l}")

    for epoch_num in range(epochs):
        for i in range(r):
            for j in range(r):
                if mat[i][j] == IGNORE_VALUE:
                    continue

                val = 0
                for k in range(c):
                    val += a[i][k] * b[k][j]

                for p in range(c):
                        
                    a[i][p] += alpha * 2 * (mat[i][j] - val) * b[p][j]
                    b[p][j] += alpha * 2 * (mat[i][j] - val) * a[i][p]

    result = np.matmul(np.asarray(a), np.asarray(b))
    return result

# Example usage
file_path = "network.csv"
graph = create_graph_from_csv(file_path)


adjacency_matrix, node_to_index, num_nodes, index_to_node = create_adjacency_matrix(graph)
result = find_svd(adjacency_matrix)
predicted = (result >= 0.3).astype(int)

print(np.sum(predicted))


output_file_path = "output_missing_links.txt"

# Perform SVD and write the result to the output file
with open(output_file_path, "w") as f:

    f.write("Original Matrix:\n")
    np.savetxt(f, adjacency_matrix, fmt='%0.2f')
    f.write("\n")


    f.write("Probability Matrix:\n")
    np.savetxt(f, result, fmt='%0.2f')
    f.write("\n")

    f.write("Predicted Matrix:\n")
    np.savetxt(f, predicted, fmt='%0.2f')
    f.write("\n")

adj_matrix_df = pd.DataFrame(adjacency_matrix, index=[index_to_node[i] for i in range(len(adjacency_matrix))], columns=[index_to_node[i] for i in range(len(adjacency_matrix))])
result_df = pd.DataFrame(result, index=[index_to_node[i] for i in range(len(result))], columns=[index_to_node[i] for i in range(len(result))])
predicted_df = pd.DataFrame(predicted, index=[index_to_node[i] for i in range(len(predicted))], columns=[index_to_node[i] for i in range(len(predicted))])

# Create an Excel file and write each DataFrame to a separate sheet
with pd.ExcelWriter('output_missing_links.xlsx') as writer:
    adj_matrix_df.to_excel(writer, sheet_name='Original Matrix')
    result_df.to_excel(writer, sheet_name='Probability Matrix')
    predicted_df.to_excel(writer, sheet_name='Predicted Matrix')

print("Output written to", output_file_path)