import networkx as nx
import csv
import itertools
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import table

def read_github():
    contributors = {}
    projects = {}
    popularity = {}
    csvreader = csv.DictReader(open('../data/combined.csv'), delimiter=',')
    for row in csvreader:
        if row['repository_fork'] == 'true':
            continue

        forks = int(row['repository_forks'])
        watchers = int(row['repository_watchers'])
        project = row['repository_url']
        popularity[project] = int(math.sqrt(forks * watchers))
        contributor = row['actor_attributes_login']
        if contributor not in contributors:
            contributors[contributor] = []
        if project not in projects:
            projects[project] = []

        contributors[contributor].append(project)
        projects[project].append(contributor)

    return contributors, projects, popularity

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


def graph(name, contributors_projects, popularity):

    contributors = contributors_projects[0]
    projects = contributors_projects[1]

    G = nx.Graph()

    for project in projects:
        project_contributors = projects[project]
        if len(project_contributors) == 1:
            G.add_node(project_contributors[0])
        else:
            for combination in itertools.combinations(projects[project], r=2):
                e0 = combination[0]
                e1 = combination[1]
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

    subgraphs = sample_graphs(nx.connected_component_subgraphs(G))
    for subgraph in subgraphs:
        if subgraph.number_of_nodes() < 3: continue

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

        graph_repos = set()
        for repos in [contributors[node] for node in subgraph.nodes()]:
            for repo in repos: graph_repos.add(repo)

        try:
            avg_popularity = np.mean([float(popularity[node]) for node in graph_repos])
        except ValueError, e:
            print "ERROR calculating popularity"
            avg_popularity = 0

        popularity_score.append(avg_popularity)

    #bins = [round(float(i)*.1,2)/2 for i in range(0,21)]
    bins = [round(float(i)*.1,1) for i in range(0,11)]

    values = [[] for _ in bins]

    idxs = np.digitize(q, bins, right=True)

    for idx,score in zip(idxs,popularity_score):
        values[idx].append(score)

    create_graph(name, bins, values)
    create_histogram(name + "-q", q, log_y_scale=True)
    create_histogram(name + "-smallworld", [x for x in small_worldness if x is not None and x < 10])
    highest_performing(name, bins, values)

    write_table("subgraphs_summary",
                table.create_table("Summary", "Summary", len(subgraphs),
                                   {
                                       "Q": q,
                                       "CC": cc,
                                       "PL": pl,
                                       "Popularity": popularity_score,
                                       "Small World": [x for x in small_worldness if x is not None],
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
    plt.hist(data)
    F = plt.gcf()
    F.savefig("../paper/images/"+filename+"-histo.png")
    plt.close(fig)

def highest_performing(filename, bins, values):
    popular_bins = []
    unpopular_bins = []

    for bin,value in zip(bins,values):
        if len(value) < 5: continue
        popular_repos = len([v for v in value if v > 6])
        unpopular_repos = len([v for v in value if v <= 1 ])
        print str(bin) + " " + str(popular_repos) + " " + str(unpopular_repos) + " " + str(len(value))
        for _ in range(int( (float(popular_repos) / len(value)) * 100 )):
            popular_bins.append(bin)
        for _ in range(int( (float(unpopular_repos) / len(value)) * 100 )):
            unpopular_bins.append(bin)

    for name, data in zip(["popular","unpopular"], [popular_bins,unpopular_bins]):
        fig = plt.figure()
        plt.hist(data, len(bins))
        F = plt.gcf()
        F.savefig("../paper/images/"+filename+"-"+name+".png")
        plt.show()
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


github_results = read_github()
graph("github", (github_results[0], github_results[1]),github_results[2] )
