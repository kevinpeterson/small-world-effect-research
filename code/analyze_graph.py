import networkx as nx
import csv
import itertools
import matplotlib.pyplot as plt
import random
import numpy as np
import MySQLdb
from scipy import stats


def read_github():
    contributors = {}
    projects = {}
    popularity = {}
    csvreader = csv.DictReader(open('../data/combined.csv'), delimiter=',')
    for row in csvreader:
        watchers = row['repository_watchers']
        project = row['repository_url']
        popularity[project] = watchers
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

def compute_random_graph(graph):
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()

    graph_r = nx.Graph()

    node_range = range(0,nodes)

    last_node = None
    for i in node_range:
        if last_node is None:
            last_node = i
        else:
            graph_r.add_edge(last_node,i)
            last_node = i

    while graph_r.number_of_edges() < edges:
        node1 = random.choice(node_range)
        node2 = random.choice(node_range)

        if node1 != node2:
            graph_r.add_edge(node1,node2)

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
    popularity_score = []

    def sample_graphs(subgraphs):
        return subgraphs

    for subgraph in sample_graphs(nx.connected_component_subgraphs(G)):
        if subgraph.number_of_nodes() < 3: continue

        graph_q = compute_q(subgraph)

        q.append(graph_q['q'])

        graph_repos = set()
        for repos in [contributors[node] for node in subgraph.nodes()]:
            for repo in repos: graph_repos.add(repo)

        try:
            avg_popularity = np.mean([float(popularity[node]) for node in graph_repos])
        except ValueError, e:
            print "ERROR calculating popularity"
            avg_popularity = 0

        popularity_score.append(avg_popularity)

    bins = [round(float(i)*.1,2)/2 for i in range(0,21)]
    #bins = [round(float(i)*.1,1) for i in range(0,11)]

    values = [[] for _ in bins]

    idxs = np.digitize(q, bins, right=True)

    for idx,score in zip(idxs,popularity_score):
        values[idx].append(score)

    create_graph(name, bins, values)
    create_histogram(name, q)
    highest_performing(name, bins, values)

def create_histogram(filename, q):
    fig = plt.figure()
    plt.yscale('log', nonposy='clip')
    plt.hist(q)
    F = plt.gcf()
    F.savefig("../paper/images/"+filename+"-histo.png")
    plt.close(fig)

def highest_performing(filename, bins, values):
    sorted_y = sorted(sum(values,[]), reverse=True)

    # throw away the low-scoring items
    highest_y = set(sorted_y[int(len(sorted_y)*.50):])

    popular_bins = []

    for bin,value in zip(bins,values):
        if len(value) < 2: continue
        percentage = len([v for v in value if v in highest_y ])
        print str(percentage) + " - " + str(len(value)) + " - " + str(float(percentage)/len(value))
        for _ in range(int( (float(percentage) / len(value)) * 100 )):
            popular_bins.append(bin)

    fig = plt.figure()
    plt.hist(popular_bins, len(bins))
    plt.show()
    F = plt.gcf()
    F.savefig("../paper/images/"+filename+"-q.png")
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

