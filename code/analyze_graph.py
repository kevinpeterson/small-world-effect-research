import networkx as nx
import csv
import itertools
import matplotlib.pyplot as plt
import random
import numpy as np
import table

def read_contributors():
    contributors = {}

    fp = file('../data/fcProjectAuthors2013-Sep.txt')
    csvreader = csv.DictReader(filter(lambda row: row[0]!='#', fp), delimiter='\t')
    for row in csvreader:
        contributor = row['author_name'].strip()
        project = row['project_id'].strip()
        if contributor not in contributors:
            contributors[contributor] = []

        contributors[contributor].append(project)
    fp.close()

    return contributors


def read_stats():
    popularity = {}

    fp = file('../data/fcProjectStats2013-Sep.txt')
    csvreader = csv.DictReader(filter(lambda row: row[0]!='#', fp), delimiter='\t')
    for row in csvreader:
        project_id = row['project_id'].strip()
        popularity_score = row['popularity_score'].strip()

        try:
            score = float(popularity_score)
        except ValueError, e:
            print "ERROR calculating project score: %s" % str(popularity_score)
            score = 0

        if project_id not in popularity: popularity[project_id] = score

    fp.close()

    return popularity


def reject_outliers(data):
    return data

def get_small_worldness(q_dict_g, q_dict_rand):
    """
    Humphries, M. D., & Gurney, K. (2008).
    Network 'small-world-ness': a quantitative method for determining canonical network equivalence. PLoS One, 3(4), e0002051.
    """
    try:
        C_g = q_dict_g['cc']
        C_rand = q_dict_rand['cc']

        L_g = q_dict_g['pl']
        L_rand = q_dict_rand['pl']

        return (C_g/C_rand) / (L_g/L_rand)
    except ZeroDivisionError:
       return None


def compute_q(graph):
    cc = nx.transitivity(graph)
    pl = nx.average_shortest_path_length(graph)

    return {"q":cc/pl,"pl":pl,"cc":cc}

def compute_random_graph(nodes, edges):

    graph_r = nx.Graph()

    node_range = range(0,nodes)

    random_edges = 0

    last_node = None
    for i in node_range:
        if last_node is None:
            last_node = i
        else:
            graph_r.add_edge(last_node,i)
            last_node = i

    while random_edges < edges:
        node1 = random.choice(node_range)
        node2 = random.choice(node_range)

        if node1 != node2:
            graph_r.add_edge(node1,node2)
            random_edges += 1

    return graph_r


def graph(name, contributors, popularity):
    G = nx.Graph()

    for contributor in contributors:
        contributor_projects = contributors[contributor]

        for combination in itertools.combinations(contributor_projects, r=2):
            e0 = combination[0]
            e1 = combination[1]
            if not G.has_edge(e0,e1) and not G.has_edge(e1,e0):
                G.add_edge(e0,e1)

    q = []
    cc = []
    pl = []
    popularity_score = []
    small_worldness = []
    subgraph_node_number = []
    subgraph_edge_number = []

    def sample_graphs(subgraphs):
        return subgraphs

    N = 0
    subgraphs = sample_graphs(nx.connected_component_subgraphs(G))
    for subgraph in subgraphs:
        N += 1

        nodes = subgraph.number_of_nodes()
        edges = subgraph.number_of_edges()

        subgraph_node_number.append(nodes)
        subgraph_edge_number.append(edges)

        graph_q = compute_q(subgraph)

        random_graph = compute_random_graph(nodes, edges)

        random_q = compute_q(random_graph)

        small_worldness.append(get_small_worldness(graph_q, random_q))

        q.append(graph_q['q'])
        cc.append(graph_q['cc'])
        pl.append(graph_q['pl'])

        try:
            avg_popularity = np.mean([float(popularity[node]) for node in subgraph])
        except ValueError, e:
            print "ERROR calculating popularity"
            avg_popularity = 0

        popularity_score.append(avg_popularity)

    print popularity_score

    bins = [round(float(i)*.1,1) for i in range(0,11)]

    values = [[] for _ in bins]

    idxs = np.digitize(q, bins, right=True)

    for idx,score in zip(idxs,popularity_score):
        values[idx].append(score)

    create_graph(name, bins, values)
    create_histogram(name + "-q", q, log_y_scale=True)
    create_histogram(name + "-smallworld", [x for x in small_worldness if x is not None])
    highest_performing(name, bins, values)

    total_projects = len(set(sum(contributors.values(), [])))

    write_table("subgraphs_summary",
                table.create_table("Summary", "fig:summary_stats", "Projects: %s, Subgraphs: %s, Contributors: %s" % (total_projects, N, len(contributors.keys())),
                                   {
                                       "Q": q,
                                       "$C^\Delta$": cc,
                                       "L": pl,
                                       "Popularity": popularity_score,
                                       "$S^\Delta$": [x for x in small_worldness if x is not None],
                                       "Nodes": subgraph_node_number,
                                       "Edges": subgraph_edge_number
                                   }))

def write_table(file_name, table_tex):
    with open("../paper/tables/"+file_name+".tex", 'w') as the_file:
        the_file.write(table_tex)

def create_histogram(filename, data, log_y_scale=False):
    fig = plt.figure()
    if log_y_scale:
        plt.yscale('log', nonposy='clip')
    plt.hist(data, color="0.75")
    F = plt.gcf()
    F.savefig("../paper/images/"+filename+"-histo.png")
    plt.close(fig)

def highest_performing(filename, bins, values):
    popular_bins = []
    unpopular_bins = []

    for bin,value in zip(bins,values):
        if len(value) < 5: continue
        popular_repos = len([v for v in value if v > 100])
        unpopular_repos = len([v for v in value if v < 50 ])
        print str(bin) + " " + str(popular_repos) + " " + str(unpopular_repos) + " " + str(len(value))
        for _ in range(int( (float(popular_repos) / len(value)) * 100 )):
            popular_bins.append(bin)
        for _ in range(int( (float(unpopular_repos) / len(value)) * 100 )):
            unpopular_bins.append(bin)

    for name, data in zip(["popular","unpopular"], [popular_bins,unpopular_bins]):
        fig = plt.figure()
        plt.hist(data, len(bins), color="0.75")
        F = plt.gcf()
        F.savefig("../paper/images/"+filename+"-"+name+".png")
        plt.close(fig)

def create_graph(filename, x, y):
    avg = [np.mean(j, axis=None) for j in y]
    median = [np.median(j) for j in y]

    fig = plt.figure()
    plt.plot(x,avg,'b-')
    plt.plot(x,median,'r-')
    F = plt.gcf()
    F.savefig("../paper/images/"+filename+"-graph.png")
    plt.close(fig)

graph("freecode", read_contributors(), read_stats())
