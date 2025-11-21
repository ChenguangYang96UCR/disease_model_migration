import csv
import argparse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx
import torch
from torch_geometric.data import Data
import os
import numpy as np

def read_csv_file(file_path, data_name = 'pd'):

    """
    Read csv file information

    Args:
        file_path (str): csv file path
        data_name (str, optional): data name, flag is true. Defaults to pd.
    """    

    with open(file_path, 'r+', encoding='utf-8') as file:
        if data_name == 'pd':
            major_feature_index = []
            Nodes = []
            Nodes_label = []
            reader = csv.reader(file)
            header = next(reader)
            header = next(reader)
            for index, feature in enumerate(header):
                if feature == 'PPE' or feature == 'DFA' or feature == 'RPDE':
                    major_feature_index.append(index)
            Node_id = 0
            Node_label = 0
            node_features = []
            for row in reader:
                if Node_id == int(row[0]):
                    node_feature = []
                    for index in major_feature_index:
                        node_feature.append(float(row[index]))
                    node_features.append(node_feature)
                    Node_label = int(row[-1])
                else:
                    Node_id = int(row[0])
                    node_features = np.array(node_features, dtype=float)
                    if len(node_features) > 0:
                        mean_features = node_features.mean(axis=0)       # mean across all rows
                    else:
                        mean_features = np.zeros(len(major_feature_index))
                    Nodes_label.append(Node_label)
                    Nodes.append(mean_features.tolist())
                    node_features = []
                    node_feature = []
                    for index in major_feature_index:
                        node_feature.append(float(row[index]))
                    node_features.append(node_feature)
                    Node_label = int(row[-1])

        elif data_name == 'als':
            major_feature_index = []
            Nodes = []
            Nodes_label = []
            reader = csv.reader(file)
            header = next(reader)
            label_index = -1
            for index, feature in enumerate(header):
                if feature != 'ID' and feature != 'ExID' and feature != 'Period' and feature != 'Subject ID' and feature != 'Death Date' \
                    and feature != 'Source' and feature != 'Survival' and feature != 'Survived' and feature != 'ALSFRS T12':
                    major_feature_index.append(index)
                if feature == 'Survival':
                    label_index = index
            Node_id = None
            Node_label = 0
            node_features = []
            for row_index, row in enumerate(reader):
                if row_index == 0:
                    Node_id = int(row[0])

                if Node_id == int(row[0]):
                    node_feature = []
                    for index in major_feature_index:
                        node_feature.append(float(row[index]))
                    node_features.append(node_feature)
                    if row[label_index] == '' : 
                        Node_label = int(0)
                    else:
                        Node_label = int(row[label_index])
                else:
                    Node_id = int(row[0])
                    node_features = np.array(node_features, dtype=float)
                    if len(node_features) > 0:
                        mean_features = node_features.mean(axis=0)       # mean across all rows
                    else:
                        mean_features = np.zeros(len(major_feature_index))
                    Nodes_label.append(Node_label)
                    Nodes.append(mean_features.tolist())

                    node_features = []
                    node_feature = []
                    for index in major_feature_index:
                        node_feature.append(float(row[index]))
                    node_features.append(node_feature)
                    if row[label_index] == '' : 
                        Node_label = int(0)
                    else:
                        Node_label = int(row[label_index])

        elif data_name == 'alz':
            major_feature_index = []
            Nodes = []
            Nodes_label = []
            reader = csv.reader(file)
            header = next(reader)
            label_index = -1
            for index, feature in enumerate(header):
                if feature != 'PatientID' and feature != 'DoctorInCharge' and feature != 'Diagnosis':
                    major_feature_index.append(index)
                if feature == 'Diagnosis':
                    label_index = index
            Node_id = None
            Node_label = 0
            node_features = []
            for row_index, row in enumerate(reader):
                if row_index == 0:
                    Node_id = int(row[0])

                if Node_id == int(row[0]):
                    node_feature = []
                    for index in major_feature_index:
                        node_feature.append(float(row[index]))
                    node_features.append(node_feature)
                    Node_label = int(row[label_index])
                else:
                    Node_id = int(row[0])
                    node_features = np.array(node_features, dtype=float)
                    if len(node_features) > 0:
                        mean_features = node_features.mean(axis=0)       # mean across all rows
                    else:
                        mean_features = np.zeros(len(major_feature_index))
                    Nodes_label.append(Node_label)
                    Nodes.append(mean_features.tolist())

                    node_features = []
                    node_feature = []
                    for index in major_feature_index:
                        node_feature.append(float(row[index]))
                    node_features.append(node_feature)
                    Node_label = int(row[label_index])

        print(f'Nodes length: {len(Nodes)}')
        print(f'Positive Nodes number {Nodes_label.count(1)}')
        print(f'Negative Nodes number {Nodes_label.count(0)}')
        return Nodes, Nodes_label
    

def create_graph_based_cosine_similarity(Nodes, similarity_matrix, node_labels, threshold = 0.95):

    """
    Using cosine similarity to generate the graph

    Args:
        Nodes (array): Nodes features X
        similarity_matrix (array): cosine similarity between each node
        node_labels (array): disease flag (0 or 1)
        threshold (float, optional): edge create threshold. Defaults to 0.95.

    Returns:
        G: networkx graph
    """    

    x = torch.tensor(Nodes, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.int)

    edge_list = []
    edge_weights = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                edge_list.append([i, j])
                edge_list.append([j, i])  # Add reverse edge for undirected graph
                edge_weights.append(similarity_matrix[i, j])
                edge_weights.append(similarity_matrix[i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    num_nodes = len(node_labels)
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True

    data = Data(x=x, 
                edge_index=edge_index, 
                # edge_attr=edge_attr, 
                y=y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data', type=str, default='pd', help='disease dataset name')
    parser.add_argument('-split', type=bool, default=False, help='disease dataset name')

    args = parser.parse_args()

    if args.data == 'pd':
        csv_file_path = 'data/pd_speech_features.csv'
        Nodes, Nodes_label = read_csv_file(csv_file_path)
    
    if args.data == 'als':
        csv_file_path = 'data/als_sample_data.csv'
        Nodes, Nodes_label = read_csv_file(csv_file_path, args.data)
        if args.split:
            split_length = int(len(Nodes) * 0.5)
            train_split_length = int(len(Nodes) * 0.1)
            train_Nodes = Nodes[:train_split_length]
            train_Nodes_label = Nodes_label[:train_split_length]
            test_Nodes = Nodes[split_length + 1:]
            test_Nodes_label = Nodes_label[split_length + 1:]
            print(f'{args.data} train node number: {len(train_Nodes)}')
            print(f'Positive train Nodes number {train_Nodes_label.count(1)}')
            print(f'Negative train Nodes number {train_Nodes_label.count(0)}')
            print(f'{args.data} test node number: {len(test_Nodes)}')
            print(f'Positive test Nodes number {test_Nodes_label.count(1)}')
            print(f'Negative test Nodes number {test_Nodes_label.count(0)}')

    if args.data == 'alz':
        csv_file_path = 'data/alzheimers_disease_data.csv'
        Nodes, Nodes_label = read_csv_file(csv_file_path, args.data)

    if args.split:
        # Pre-train data
        train_similarity_matrix = cosine_similarity(train_Nodes)
        train_data = create_graph_based_cosine_similarity(train_Nodes, train_similarity_matrix, train_Nodes_label)
        
        os.makedirs(f'datasets/{args.data}', exist_ok=True)
        torch.save(train_data, f'datasets/{args.data}/train_{args.data}.pt')

        # Transfer data
        test_similarity_matrix = cosine_similarity(test_Nodes)
        test_data = create_graph_based_cosine_similarity(test_Nodes, test_similarity_matrix, test_Nodes_label)
        
        os.makedirs(f'datasets/{args.data}', exist_ok=True)
        torch.save(test_data, f'datasets/{args.data}/test_{args.data}.pt')
    else:
        similarity_matrix = cosine_similarity(Nodes)
        data = create_graph_based_cosine_similarity(Nodes, similarity_matrix, Nodes_label)
        
        os.makedirs(f'datasets/{args.data}', exist_ok=True)
        torch.save(data, f'datasets/{args.data}/{args.data}.pt')

    # print(f"Graph density: {networkx.density(G):.4f}")





