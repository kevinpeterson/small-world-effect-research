import networkx as nx
import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import table

def read_contributors():
    contributors = {}
    projects = {}

    fp = file('../data/fcProjectAuthors2013-Sep.txt')
    csvreader = csv.DictReader(filter(lambda row: row[0]!='#', fp), delimiter='\t')
    for row in csvreader:
        contributor = row['author_name'].strip()
        project = row['project_id'].strip()
        if contributor not in contributors:
            contributors[contributor] = []
        if project not in projects:
            projects[project] = []

        contributors[contributor].append(project)
        projects[project].append(contributor)
    fp.close()

    return contributors,projects


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

def get_q(q_dict_g, q_dict_rand):
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


def compute_stats(graph):
    cc = nx.transitivity(graph)
    pl = nx.average_shortest_path_length(graph)
    avg_cc = nx.average_clustering(graph, weight="weight")

    return {"pl":pl,"cc":cc, "avg_cc": avg_cc}

def compute_random_graph(nodes, edges, g):

    degrees = sum(g.degree().values())
    avg_degree = (float(degrees)/float(nodes))
    try:
        return nx.connected_watts_strogatz_graph(nodes, int(avg_degree), 0.5)
    except nx.NetworkXPointlessConcept:
        assert (edges == (nodes - 1))
        return g

def graph(name, data, popularity):
    contributors = data[0]
    projects = data[1]
    G = nx.Graph()

    for project in projects:
        project_contributors = projects[project]

        for combination in itertools.combinations(project_contributors, r=2):
            e0 = combination[0]
            e1 = combination[1]
            if not G.has_edge(e0,e1) and not G.has_edge(e1,e0):
                G.add_edge(e0,e1, weight=1)
            else:
                G[e0][e1]['weight'] += 1

    q = []
    cc = []
    pl = []
    avg_cc = []
    popularity_score = []
    subgraph_node_number = []
    subgraph_edge_number = []

    def sample_graphs(subgraphs):
        return subgraphs

    N = 0
    subgraphs = sample_graphs(nx.connected_component_subgraphs(G))
    for subgraph in subgraphs:
        nodes = subgraph.number_of_nodes()
        edges = subgraph.number_of_edges()

        if nodes < 3: continue

        subgraph_node_number.append(nodes)
        subgraph_edge_number.append(edges)

        graph_stats = compute_stats(subgraph)

        random_graph = compute_random_graph(nodes, edges, subgraph)

        random_stats = compute_stats(random_graph)

        q_value = get_q(graph_stats, random_stats)

        if q_value is None: continue

        N += 1

        q.append(q_value)

        cc.append(graph_stats['cc'])
        pl.append(graph_stats['pl'])
        avg_cc.append(graph_stats['avg_cc'])

        graph_projects = set(sum([contributors[j] for j in subgraph],[]))

        try:
            avg_popularity = np.mean([float(popularity[node]) for node in graph_projects])
        except ValueError:
            print "ERROR calculating popularity"
            avg_popularity = 0

        popularity_score.append(avg_popularity)

    q_bins = np.linspace(min(q),max(q), num=8)
    q_values = [[] for _ in q_bins]

    q_idxs = np.digitize(q, q_bins, right=True)

    for idx,score in zip(q_idxs,popularity_score):
        q_values[idx].append(score)

    create_graph(name, q_bins, q_values)
    create_histogram(name + "-q", "$Q$", q, log_y_scale=True)
    highest_performing(name, q_bins, q_values)

    total_projects = len(set(sum(contributors.values(), [])))

    write_table("subgraphs_summary",
                table.create_table("Summary", "fig:summary_stats", "Projects: %s, Subgraphs: %s, Contributors: %s" % (total_projects, N, len(contributors.keys())),
                                   {
                                       "$Q$": q,
                                       "$C$": cc,
                                       "$\overline{C}$": avg_cc,
                                       "$L$": pl,
                                       "Popularity": popularity_score,
                                       "Nodes": subgraph_node_number,
                                       "Edges": subgraph_edge_number
                                   }))

def write_table(file_name, table_tex):
    with open("../paper/tables/"+file_name+".tex", 'w') as the_file:
        the_file.write(table_tex)

def create_histogram(filename, x_label, data, log_y_scale=False):
    fig = plt.figure()
    if log_y_scale:
        plt.yscale('log', nonposy='clip')
    plt.hist(data, color="0.75")
    plt.xlabel(x_label)
    plt.ylabel("projects")
    F = plt.gcf()
    F.savefig("../paper/images/"+filename+"-histo.png")
    plt.close(fig)

def highest_performing(filename, bins, values):
    popular_bins = []
    unpopular_bins = []

    for bin,value in zip(bins,values):
        if len(value) < 3: continue
        popular_repos = len([v for v in value if v > 100])
        unpopular_repos = len([v for v in value if v < 100 ])
        print str(bin) + " " + str(popular_repos) + " " + str(unpopular_repos) + " " + str(len(value))
        for _ in range(int( (float(popular_repos) / len(value)) * 100 )):
            popular_bins.append(bin)
        for _ in range(int( (float(unpopular_repos) / len(value)) * 100 )):
            unpopular_bins.append(bin)

    for name, data in zip(["popular","unpopular"], [popular_bins,unpopular_bins]):
        fig = plt.figure()
        plt.hist(data, 4, color="0.75")
        plt.xlabel("$Q$")
        plt.ylabel("% " + name)
        F = plt.gcf()
        F.savefig("../paper/images/"+filename+"-"+name+".png")
        plt.close(fig)

def create_graph(filename, x, y):
    for j in y: print len(j)
    avg = [np.mean(j, axis=None) for j in y]
    median = [np.median(j) for j in y]

    fig = plt.figure()
    plt.plot(x,avg,'-')
    plt.plot(x,median,'--')
    plt.xlabel("$Q$")
    plt.ylabel("popularity")
    F = plt.gcf()
    F.savefig("../paper/images/"+filename+"-graph.png")
    plt.close(fig)

graph("freecode", read_contributors(), read_stats())
