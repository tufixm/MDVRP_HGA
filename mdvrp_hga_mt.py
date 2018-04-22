### Sparsity RD Test - MDVRP GA ###
### Mihnea Tufis ###

import pandas as pd
import numpy as np
import random
import bisect
import itertools as itt
from scipy.cluster.vq import kmeans2


# simulate the flip of a coin with probability p
# will assist in deiding whether to apply genetic operations in the current iteration
def do_operation(p):
    return random.random() < p


# compute savings matrix for the Clarke-Wright algo for the link associated to depot_id
# return a list of the savings for each pair, sorted in decreasing order
# e.g. [((5, 6), 13.342367637579066), ...]
def get_cws_matrix(depot_id):
    depot_pos = np.array(depots[depot_id][:2])
    savings = {}
    # iterate over 2-nodes combinations
    for pair in itt.combinations(groups[depot_id], r=2):
        customer_1 = customers[pair[0]]
        cpos_1 = np.array(customer_1[:2])
        customer_2 = customers[pair[1]]
        cpos_2 = np.array(customer_2[:2])
        savings[pair] = np.linalg.norm(depot_pos-cpos_1) + np.linalg.norm(depot_pos-cpos_2) - np.linalg.norm(cpos_1-cpos_2)
    return sorted(list(savings.items()), key = lambda x: x[1], reverse = True)


# savings - a list of pairs of customers, in decreasing order of their savings value
# link - the group of customers being routed
# returns a list of 
def cws(savings, link_id):
    assigned = {customer: -1 for customer in groups[link_id]}
    routes_capacity = []
    route_no = -1
    for pair, s in savings:
        c1 = pair[0]
        c2 = pair[1]
        capacity_c1 = customers[c1][2]
        capacity_c2 = customers[c2][2]
        if assigned[c1] == -1 and assigned[c2] == -1 and capacity_c1+capacity_c2 <= t_capacity:
            route_no = len(routes_capacity)
            assigned[c1] = route_no
            assigned[c2] = route_no
            routes_capacity.append(capacity_c1 + capacity_c2)
        elif assigned[c1] != -1 and assigned[c2] == -1 and routes_capacity[assigned[c1]] + capacity_c2 <= t_capacity:
            routes_capacity[assigned[c1]] += capacity_c2
            assigned[c2] = assigned[c1]
        elif assigned[c2] != -1 and assigned[c1] == -1 and routes_capacity[assigned[c2]] + capacity_c1 <= t_capacity:
            routes_capacity[assigned[c2]] += capacity_c1
            assigned[c1] = assigned[c2]
        elif assigned[c1] != assigned[c2] and assigned[c1] != -1 and assigned[c2] != -1 and routes_capacity[assigned[c1]] + routes_capacity[assigned[c2]] <= t_capacity:
            expanding_route, other_route = min(assigned[c1], assigned[c2]), max(assigned[c1], assigned[c2])
            routes_capacity[expanding_route] = routes_capacity[assigned[c1]] + routes_capacity[assigned[c2]]
            routes_capacity[other_route] = -1
            for key, value in assigned.items():
                if value == other_route:
                    assigned[key] = expanding_route
        # Tricky with this early stop, because I can miss: 1. merging sub-capacity routes
        # early stop if all routes have reached capacity
        #if all(rc >= 12 or rc == -1 for rc in routes_capacity) and all(asgn_route != -1 for asgn_route in list(assigned.values())):
        #    return assigned
    for key, value in assigned.items():
        single_route_no = len(routes_capacity)
        if value == -1:
            assigned[key] = single_route_no
            routes_capacity.append(customers[key][2])
    return assigned


# get groups of customers around depots
# returns a dict of the form {'0A': [1,2,3,4,5,6], '0B': [7,8,9,10,11,12]}
def get_groups(customers, depots):
    # prepare for clustering
    customers_coord = [list(map(float, v[:2])) for v in customers.values()]
    # depots are processed by key, in ascending order; it depends on the key type, whether this order is lexicographic or numeric
    depots_coord = [list(map(float, depots[key][:2])) for key in depots_names]
    # restrinc k-means to only 1 iteration, with preset centroids (depots); pick up only the class labels
    _, labels = kmeans2(data=customers_coord, iter=1, k=depots_coord, minit='matrix')
    # iterate through the labels and assign each
    # alternatively, can be done with the vq function; but that one is better to assign entire observations (easier to work with indices in this case)
    groups = {}
    for obs_idx in range(len(labels)):
        groups.setdefault(depots_names[labels[obs_idx]], []).append(obs_idx+1)
    return groups


# returns time spent to make a delivery from depot to an associated route
# sum of distances between: (depot -> customer 1) + two adjacent scheduled customers + (last customer -> depot)
def get_sched_dist(sched, depot):
    depot_pos = np.array(depot[:2])
    first_pos = np.array(customers[sched[0]][:2]) # first scheduled customer
    last_pos = np.array(customers[sched[-1]][:2]) # last scheduled customer
    total_dist = np.linalg.norm(depot_pos-first_pos) + np.linalg.norm(last_pos-depot_pos)
    # pairs of consecutive customers in a scheduled route
    for c1, c2 in zip(sched, sched[1:]):
        c1_pos = np.array(customers[c1][:2])
        c2_pos = (customers[c2][:2])
        total_dist += np.linalg.norm(c1_pos-c2_pos)
    return total_dist


# compute the time for one specific depot (over all routes)
# link: a link in a chromosome in the form of a tuple ('0A', [[1,2,4], [3,5,6]])
def time_depot(link, depot_id):
    return sum(get_sched_dist(route, depots[depot_id]) for route in link)


# receives the depot 
# takes its associated group e.g., [1,2,3,4,5,6]
# returns a list of the routes in the given group e.g., [[1,2,4], [3,5,6]] 
def get_routes(depot_id):
    ordered_savings = get_cws_matrix(depot_id)
    route_partition = cws(ordered_savings, depot_id)
    routes = {}
    for customer, route_no in route_partition.items():
        routes.setdefault(route_no, []).append(customer)
    return [route for route in list(routes.values())]



# rest: remaining customer nodes (route) to be scheduled
# schedule: the schedule list
# recursively builds the schedule from a route
def compose_schedule(rest, sched):
    if rest == []:
        return sched
    nearest_node = -1
    nearest_dist = None
    last = sched[-1]
    for dest in rest:
        last_pos = np.array(customers[last][:2])
        dest_pos = np.array(customers[dest][:2])
        d = np.linalg.norm(last_pos-dest_pos)
        if nearest_dist is None:
            nearest_dist = d
            nearest_node = dest
        elif d < nearest_dist:
            nearest_dist = d
            nearest_node = dest
    rest.remove(nearest_node)
    sched.append(nearest_node)
    return compose_schedule(rest, sched)


## IMPROVEMENT ON THE NN HEURISTIC - untested properly; just an idea ##
## schedule the closest two nodes in the route as first and last to be served
## apply NN simultaneously from both sides of the list, starting with these 2
def nearest_node2(node, rest):
    nearest_node = -1
    nearest_dist = None
    node_pos = np.array(customers[idx][:2])
    for next_node in rest:
        next_pos = np.array(customers[next_node][:2])
        d = np.linalg.norm(node_pos-next_pos)
        if nearest_dist is None or d < nearest_dist:
            nearest_dist = d
            nearest_node = next_node
    return nearest_node

def compose_schedule2(route, depot_id):
    # initialize a list of size route with -1s
    schedule = [-1]*len(route)
    rest = route[:]
    depot_pos = np.array(depots[depot_id][:2])
    distances = [np.linalg.norm(depot_pos - np.array(customers[node_id][:2])) for node_id in route]
    sorted_dist = sorted(zip(route, distances), key = lambda x: x[1])
    schedule[0] = sorted_dist[0][0]
    schedule[-1] = sorted_dist[1][0]
    rest.remove(schedule[0])
    rest.remove(schedule[-1])
    for idx in range(1, len(route)//2):
        nn = nearest_node2(route[idx-1], rest)
        if nn != -1:
            schedule[idx] = nn
            rest.remove(nn)
        nn = nearest_node(route[len(route)-1-idx])
        if nn != -1:
            schedule[len(route)-1]
            rest.remove(nn)
    if schedule[len(route)//2 + 1] == -1:
        schedule[len(route)//2 + 1] = rest[0]
    return schedule

### end of IMPROVEMENT PROPOSAL ###
        


### GA SPECIFIC METHODS ###

# transforms a chromosome from its [1,2,3,...] notation, into a dictionary notation encoding routes and schedules
# this is invoked when evaluating newly generated chromosomes, for which the score is computed first at route level
def transform(chromosome):
    t_chromosome = {}
    link_id = 0
    for lp1, lp2 in zip(links_pos, links_pos[1:]):
        link = chromosome[lp1:lp2]
        # (customer, costs) list for the customers in the given link
        link_costs = [(key, customers[key][2]) for key in link]
        # cumulative costs for a link
        cum_costs = list(itt.accumulate([e[1] for e in link_costs]))
        # divide by the transport capacity and use the rest to assign to a single route
        routes = [cost//(t_capacity+1) for cost in cum_costs]
        schedules = {}
        for r, c in zip(routes, link):
            schedules.setdefault(r, []).append(c)
        t_chromosome[depots_names[link_id]] = []
        for r in sorted(schedules):
            t_chromosome[depots_names[link_id]].append(schedules[r])
        link_id+=1
    return t_chromosome


# returns the max delivery time over all of the chromosome's links
# receives a chromosome in its transformed form (i.e. w/routes and schedules)
def evaluate(t_chromosome):
    return max(time_depot(link, depot_id) for depot_id, link in t_chromosome.items())


# applies the roulette selection method over the population
# returns the number of the selected chromosome
def selection():
    roulette_throw = random.uniform(0,1)
    roulette_sel = bisect.bisect_left(cum_chrom_proba, roulette_throw)
    return chrom_proba[roulette_sel][1]


# get one xover offspring at a time
def get_xover_offspring(parent, substr, start):
    seq = [gene for gene in parent if gene not in substr]
    return seq[:start] + substr + seq[start:]


# order crossover of two parents
# returns a list of the 2 expected offsprings; calls get_xover_offspring
def crossover(parent_1, parent_2):
    len_p1 = len(parent_1)-1
    #start and stop positions; stop > start; avoid crossing over by selecting subsequences of only 1 gene
    start = random.randint(0,len_p1-2)   
    stop = random.randint(start+2,len_p1)
    # return the offsprings
    # copy [start:pos] from parent1 to offspring; fill in the rest of the offspring with unused genes form parent 2
    # repeat with inverted parents
    offspring_1 = get_xover_offspring(parent_2, parent_1[start:stop], start)
    offspring_2 = get_xover_offspring(parent_1, parent_2[start:stop], start)
    return [offspring_1, offspring_2]


# heuristic mutation
# uses the permutations of 3 genes
# returns the 5 resulting permutations 
def mutate_heu(parent):
    offspring_list = []
    # randomly select 3 gene positions
    positions = random.sample(range(len(parent)),3)
    permutations = list(itt.permutations(positions))
    for permutation in permutations:
        offspring = parent[:]
        if list(permutation) != positions:
            for (x,y) in list(zip(positions, list(permutation))):
                offspring[x] = parent[y]
            offspring_list.append(offspring)
    return offspring_list


# inversion mutation
def mutate_inv(parent):
    len_p = len(parent)-1
    #start and stop (exclusive) positions
    #stop > start + 2 to avoid reverting a 1-gene list and generating an identical offspring
    start = random.randint(0,len_p-2)   
    stop = random.randint(start+2,len_p)
    offspring = parent[:start] + parent[start:stop][::-1] + parent[stop:]
    # ^ faster alternative for the reversed part would be list(reversed(parent[start:stop]))
    return offspring


# returns one permuted offspring and its iteration swaps
def generate_isp_offsprings(parent, positions, permutation):
    offspring_base = parent[:]
    # assign the permuted positions in the base offspring
    for (x,y) in list(zip(positions, permutation)):
        offspring_base[x] = parent[y]
    offsprings = [offspring_base]
    for position in positions:
        # left offspring; from base, then swap
        # don't swap left the first element
        # if you do, it will swap it with the last
        if position > 0:
            offspring_l = offspring_base[:]
            offspring_l[position-1], offspring_l[position] = offspring_l[position], offspring_l[position-1]
            offsprings.append(offspring_l)
        # right offspring; from base, then swap
        # don't swap right the last element (only introduces a copy of the same solution)
        if position < len(parent)-1:
            offspring_r = offspring_base[:]
            offspring_r[position], offspring_r[position+1] = offspring_r[position+1], offspring_r[position]
            offsprings.append(offspring_r)
    return offsprings


# returns the ISP swaps of parent, for the links described in links_pos
# calls generate_isp_offsprings
def isp_swap(parent, links_pos):
    positions = []
    for link_1, link_2 in zip(links_pos, links_pos[1:]):
        # generate a random gene position, from each link
        # e.g. from [0,2], [3,4], [5,end]
        rand_gene_pos = random.randint(link_1, link_2-1)
        positions.append(rand_gene_pos)
    # generate all permutations of selected genes
    permutations = list(itt.permutations(positions))
    while True:
        rand_perm_pos = random.randint(0,len(permutations)-1)
        # choose one of these to perform ISP on it
        permutation = list(permutations[rand_perm_pos])
        # make sure we don't use the initial permutation
        if permutation != positions:
            return generate_isp_offsprings(parent, positions, permutation)


#################
### MAIN PART ###
#################

# GA parameters
pop_size = 25
xover_rate = 0.4
mut_rate = 0.2
xover_pairs = 5
iter_no = 5
mutation_size = 5 # how many chromosomes could receive either mutation following xover
population = []
pop_fitness = []
init_parent = []
links_pos = []


# represent the data as dictionaries {k: (x, y, L)}
# k = id of the node; x,y = 2D coordinates of a customer; L = load demand of a customer
# same applies for depots, just that L represents the capacity of a depot
# READ DATA FROM A FILE OR A DB
# store it into customers and depots
# customers = {}
# depots = {}
# for this example I am using only the stuff from the article
customers = {1:(10,2,6), 2:(10,6,3), 3:(10,10,4), 4:(12,4,3), 5:(12,7,4), 6:(12,9,4), 7:(14,3,7), 8:(14,5,3), 9:(14,12,5), 10:(16,1,2), 11:(16,6,4), 12:(16,9,3)}
depots = {'0A':(5,5,12), '0B':(20,5,12)}
t_capacity = 12 # max capacity of a truck / route
depots_names = sorted(list(depots.keys()))
vehicle_speed = 50

# initialization => first chromosome
groups = get_groups(customers, depots)
init_parent = []
links_pos = [0]
routes = {}
for group_key in list(groups.keys()):
    group = groups[group_key]
    links_pos.append(links_pos[-1] + len(group))
    print(group_key, routes)
    routes[group_key] = get_routes(group_key)
    print(routes)
    for route in routes[group_key]:
        start_node = route[random.randint(0, len(route)-1)]
        # remove destroys the initial route, by popping out each node once included in a schedule
        # if these initial routes are needed, we need to replace the use of remove with del or pop
        # for now they're not, since they will appear in the init_parent anyway
        route.remove(start_node)
        route_schedule = compose_schedule(route, [start_node])
        init_parent += route_schedule

population.append(init_parent)
population += isp_swap(init_parent, links_pos)
pop_fitness = [evaluate(transform(chromosome)) for chromosome in population]

# HGA
for generation in range(iter_no):
    print('Generation: ', generation)

    offsprings = []
    # assess the fitness of the population in the current generation
    total_fitness = sum(pop_fitness)
    # compute the probabilities attached to each chromosome in the population
    chrom_proba = [(chrom_fit)/(total_fitness) for chrom_fit in pop_fitness]
    # this will sort them in an ordered list of tuples (Ph, Xh), where Ph is the proba of chromosome Xh
    # and the second is the chromosome idx in the parent population
    chrom_proba = sorted((proba, idx) for idx, proba in enumerate(chrom_proba))
    # and the cumulative probas of the ordered fitnesses
    cum_chrom_proba = list(itt.accumulate([e[0] for e in chrom_proba]))
    
    #print('Probas', chrom_proba)
    #print('Cumulative probas: ', cum_chrom_proba)
    
    print('Population at start of generation')
    for c in population:
        print(c)
    
    # attempt crossover for a predefined number of pairs
    for pair in range(xover_pairs):
        parent_1 = population[selection()]
        # select a second parent different from the first one
        while True:
            parent_2 = population[selection()]
            if parent_1 != parent_2:
                break
        
        # decide if xover is happening; if not, just transfer the parents to the new generation
        if do_operation(xover_rate):
            x_offsprings = crossover(parent_1, parent_2)
            if x_offsprings[0] not in offsprings:
                offsprings.append(x_offsprings[0])
            if x_offsprings[1] not in offsprings:
                offsprings.append(x_offsprings[1])
        else:
            if parent_1 not in offsprings:
                offsprings.append(parent_1)
            if parent_2 not in offsprings:
                offsprings.append(parent_2)
    
    mutation_size = len(offsprings)//2
    # attempt mutations; half the times the heuristic m. and the other half the inversion m.
    mut_heu_ids = random.sample(range(0,len(offsprings)-1), mutation_size) # randomly pick mutation_size offspring indices for heuristic mutation
    mut_inv_ids = [x for x in range(0,len(offsprings)) if x not in set(mut_heu_ids)] # and the rest of them for inversion mutation
    
    for mut_idx in mut_heu_ids:
        chromosome = offsprings[mut_idx]
        if do_operation(mut_rate):
            mut_offsprings = [o for o in mutate_heu(chromosome) if o not in offsprings]
            offsprings += mut_offsprings
    
    for idx in mut_inv_ids:
        chromosome = offsprings[mut_idx]
        if do_operation(mut_rate):
            mut_offspring = mutate_inv(chromosome)
            if mut_offspring not in offsprings:
                offsprings.append(mutate_inv(chromosome))
    
    # improve solutions with ISP
    for idx in range(len(offsprings)):
        while True:
            swaps = isp_swap(offsprings[idx], links_pos)
            swap_fit = [evaluate(transform(chromosome)) for chromosome in swaps]
            max_idx, max_val = max(enumerate(swap_fit), key = lambda x: x[1])
            # if the best fit swap offspring is not better than its parent, stop
            if max_val < evaluate(transform(offsprings[idx])):
                break
            elif swaps[max_idx] not in offsprings: # otherwise replace the parent with the fittest offspring and re-swap
                offsprings[idx] = swaps[max_idx]
    
    # sort the population, based on their fitness, in descending order           
    pop_and_fit = sorted([(o, evaluate(transform(o))) for o in offsprings], key = lambda x: x[1], reverse = True)
    population, pop_fitness = zip(*pop_and_fit)
    # and retain only the first pop_size most fit; if less than pop_size, retain all
    population = list(population)[:min(pop_size, len(population))]
    pop_fitness = list(pop_fitness)[:min(pop_size, len(population))]

# end of the HGA process; select the fittest chromosome and schedule it

solution = population[0]
print(transform(solution))


### END ###





