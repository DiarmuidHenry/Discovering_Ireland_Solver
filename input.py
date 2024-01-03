import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
import sys
import array
from timeit import default_timer as timer

# Start timer, to see how long running the code takes
start = timer()

# Read the spreadsheet of paths to neighbouring towns into a pandas DataFrame
df = pd.read_csv("sym.csv", header=None)  

# Convert the DataFrame to a NumPy array
edge_weights_matrix = df.values

if edge_weights_matrix.shape[0] != edge_weights_matrix.shape[1]:
    print("Input data not a square spreadsheet. Please check, fix and rerun program")
else:

    # Getting number of town cards from size of edge_weight_matrix

    # Now, edge_weights_matrix contains the numerical values from the spreadsheet as a NumPy array

    # Construct list of all cards, labelled by their corresponding numbers
    number_of_towns = edge_weights_matrix.shape[0]
    all_cards = list(range(1, number_of_towns + 1));

    # List of all entry/exit cards. Must be manually enterred
    entry_cards = [
        5, 9, 31, 39, 47, 50
    ];

    # List of all town cards, which is all_cards with entry_cards removed
    town_cards = [i for i in all_cards if i not in entry_cards]

    # Setting range of number of town cards
    min_town_cards = 8
    max_town_cards = 8

    # Randomly choosing number of town cards from given range
    number_of_town_cards = random.randint(min_town_cards, max_town_cards);

    # Assigning town cards
    assigned_town_cards = random.sample(town_cards, number_of_town_cards);
    # Fix town cards when testing
    # assigned_town_cards = [6, 17, 19]
    print("Assigned Town Cards are:")
    print(assigned_town_cards)

    # Assigning entry/exit cards, allowing for both to be the same
    assigned_entry_cards = random.choices(entry_cards, k=2);
    # Fix entry cards when testing
    # assigned_entry_cards = [5, 5]
    print("Assigned Entry Cards are:")
    print(assigned_entry_cards)

    # Combining the above to give the dealt hand
    dealt_hand = np.hstack((assigned_entry_cards, assigned_town_cards))
    print("Dealt hand is:")
    print(dealt_hand)

    # Create the graph G with nodes from town_cards and entry_cards, as well as the edge weights defined in sym.csv
    G = nx.from_numpy_array(edge_weights_matrix);

    # Relabel nodes in G, only relevant for physical plotting of graph
    G = nx.convert_node_labels_to_integers(G, 1)

    # Checking the graph is the right size
    # print("Number of nodes in G:")
    # print(G.number_of_nodes());
    # print("Number of edges in G:");
    # print(G.number_of_edges());

    # Checking the shortest path length/weight function on a path between 2 nodes
    # a = (random.choice(all_cards));
    # b = (random.choice(all_cards));
    # print(a,b);
    # print("Shortest path from", a, "to", b, ":");
    # print(nx.shortest_path(G, a, b, weight="weight"));
    # print("Path length:");
    # print(nx.shortest_path_length(G, a, b, weight="weight"))

    # Draw graph representation of G
    # layout = nx.spring_layout(G)
    # nx.draw(G, layout);
    # labels = nx.get_edge_attributes(G, "weight");
    # nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels);
    # nx.draw_networkx_labels(G, pos=layout)
    # plt.show();

    # Lists all pairs of nodes; length of path between them; route taken.
    distances = []
    all_shortest_paths = []
    for i in G.nodes :
        for j in G.nodes:
            all_shortest_paths.append(nx.shortest_path(G, source=i, target=j, weight="weight"));
            distances.append(nx.shortest_path_length(G, i, j, weight="weight"));

    # Reshape distances into a square array, easier to find relevant distances using indices
    distances = np.reshape(distances, newshape=(len(all_cards),len(all_cards)));
    # print(distances)
    # print(len(all_shortest_paths))
    # print(all_shortest_paths)


    # print(distances.shape[0])

    # print(distances.shape[1])

    # Output the data from distances to a csv file
    dataframe = pd.DataFrame(distances) 
    dataframe.to_csv(r"distances.csv")

    # List all permutations of assigned_town_cards
    possible_town_routes = list(permutations(assigned_town_cards));
    # print(len(possible_town_routes));

    # Create start and end card arrays, 
    start_entry = np.full((len(possible_town_routes), 1), assigned_entry_cards[0])
    end_entry = np.full((len(possible_town_routes), 1), assigned_entry_cards[1])

    # Stacking the entry and and card arrays with possible routes through town cards, we get all possible full routes
    all_possible_routes = np.hstack((start_entry, possible_town_routes, end_entry))

    # print("all possible routes:")
    # print(all_possible_routes)

    # print(all_possible_routes.shape[0])

    # print(all_possible_routes.shape[1])

    # Calculate total route length for all possible routes
    route_lengths = []
    for i in range(0, all_possible_routes.shape[0]-1):
        for j in range(0, all_possible_routes.shape[1]-1):
            route_lengths.append(distances[ all_possible_routes[i,j]-1, all_possible_routes[i,(j+1)]-1 ])

    # Reshaping route lengths into an array, one row for each route. Each number represents distance from one town ot the next
    route_lengths = np.reshape(route_lengths, newshape=((all_possible_routes.shape[0]-1), all_possible_routes.shape[1]-1));
    # print(route_lengths)

    # Summing these numbers together to get the route length for each possible route
    route_lengths = np.sum(route_lengths,axis=1)
    # print(route_lengths)

    # Finding the minimum route length, and its corresponding index
    min_length = np.min(route_lengths)
    min_indices = [i for i, x in enumerate(route_lengths) if x == min_length]

    # print(min_indices)

    print("Shortest route/s for dealt cards:")

    # Locating the shortest route/s in all_possible_routes from the indix/indicies above, shown by the order in which the assigned town cards should be visited.
    routes_to_take = [];
    for i in range(0, (len(min_indices))):
        routes_to_take.append(all_possible_routes[min_indices[i]]);

    routes_to_take = np.asarray(routes_to_take);
    print(routes_to_take)

    # Printing the route length for these routes. As the route length is the same for all instances, only the first needs printing
    print("Route length:")
    print(route_lengths[min_indices[0]]);

    # Get a more detailed list of route to follow, by showing all towns visited in between assigned town cards from all_shortest_paths and routes_to_take. Tricky part is getting correct index in the list all_shortest_paths
    # detailed_routes_to_take = assigned_entry_cards[0];
    lists = [[assigned_entry_cards[0]] for _ in range(0,len(min_indices))]
    for i in range(0, len(min_indices)):
        for j in range(0, len(routes_to_take[i])-1):
            # .copy() is used here so that no changes are made to all_shortest_paths.
            next = all_shortest_paths[(routes_to_take[i][j] - 1)*len(all_cards) + routes_to_take[i][j+1] - 1].copy();
            next.pop(0);
            lists[i] += next;

    # Removing duplicate lists, in the instance that 2 different ways of visiting town cards ends in the same detailed route. E.g. 39,40,45,33 and 39,45,40,33
    for i in range(0, len(min_indices)-1):
        for j in range(0, len(min_indices)-1):
            if i!=j and lists[i]==lists[j]:
                lists.remove(lists[i])

    print("Corresponding detailed route/s:")
    for i in range(0, len(lists)):
        print(lists[i])

    end = timer()

    # Time taken shown in seconds to 5sf
    time_taken = round((end - start), 5)

    print("Time taken:")
    print(time_taken, "seconds")