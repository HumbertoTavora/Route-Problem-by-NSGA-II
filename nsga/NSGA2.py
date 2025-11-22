
import os
import io
import random
import numpy
import fnmatch
import csv
import array

from csv import DictWriter
from json import load, dump
from deap import base, creator, tools, algorithms, benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume


BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def load_instance(json_file):
    """
    Inputs: path to json file
    Outputs: json file object if it exists, or else returns NoneType
    """
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
    return None

def routeToSubroute(individual, instance):
    """
    Inputs: Sequence of customers that a route has
            Loaded instance problem
    Outputs: Route that is divided in to subroutes
             which is assigned to each vechicle.
    """
    route = []
    sub_route = []
    vehicle_load = 0
    last_customer_id = 0
    vehicle_capacity = instance['vehicle_capacity']
    
    for customer_id in individual:
        demand = instance[f"customer_{customer_id}"]["demand"]
        updated_vehicle_load = vehicle_load + demand

        if(updated_vehicle_load <= vehicle_capacity):
            sub_route.append(customer_id)
            vehicle_load = updated_vehicle_load
        else:
            route.append(sub_route)
            sub_route = [customer_id]
            vehicle_load = demand
        
        last_customer_id = customer_id

    if sub_route != []:
        route.append(sub_route)
    return route


def printRoute(route, merge=False):
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


def getNumVehiclesRequired(individual, instance):
    """
    Inputs: Individual route
            Json file object loaded instance
    Outputs: Number of vechiles according to the given problem and the route
    """
    updated_route = routeToSubroute(individual, instance)
    num_of_vehicles = len(updated_route)
    return num_of_vehicles


def getRouteCost(individual, instance, unit_cost=1):
    """
    Inputs : 
        - Individual route
        - Problem instance, json file that is loaded
        - Unit cost for the route (can be petrol etc)

    Outputs:
        - Total cost for the route taken by all the vehicles
    """
    total_cost = 0
    updated_route = routeToSubroute(individual, instance)

    for sub_route in updated_route:
        sub_route_distance = 0
        last_customer_id = 0

        for customer_id in sub_route:
            distance = instance["distance_matrix"][last_customer_id][customer_id]
            sub_route_distance += distance
            last_customer_id = customer_id

        sub_route_distance = sub_route_distance + instance["distance_matrix"][last_customer_id][0]

        sub_route_transport_cost = unit_cost*sub_route_distance

        total_cost = total_cost + sub_route_transport_cost
    
    return total_cost


def eval_indvidual_fitness(individual, instance, unit_cost):
    """
    Inputs: individual route as a sequence
            Json object that is loaded as file object
            unit_cost for the distance 
    Outputs: Returns a tuple of (Number of vechicles, Route cost from all the vechicles)
    """

    vehicles = getNumVehiclesRequired(individual, instance)

    route_cost = getRouteCost(individual, instance, unit_cost)

    return (vehicles, route_cost)



def cxOrderedVrp(input_ind1, input_ind2):

    ind1 = [x-1 for x in input_ind1]
    ind2 = [x-1 for x in input_ind2]
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    ind1 = [x+1 for x in ind1]
    ind2 = [x+1 for x in ind2]
    return ind1, ind2


def mutationShuffle(individual, indpb):
    """
    Inputs : Individual route
             Probability of mutation betwen (0,1)
    Outputs : Mutated individual according to the probability
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = \
                individual[swap_indx], individual[i]

    return individual,


def createStatsObjs():
    """
    Inputs : None
    Outputs : tuple of logbook and stats objects.
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "Generation", "evals", "avg", "std", "min", "max", "best_one", "fitness_best_one"
    return logbook, stats


def recordStat(invalid_ind, logbook, pop, stats, gen):
    """
    Inputs : invalid_ind - Number of children for which fitness is calculated
             logbook - Logbook object that logs data
             pop - population
             stats - stats object that compiles statistics
    Outputs: None, prints the logs
    """
    record = stats.compile(pop)
    best_individual = tools.selBest(pop, 1)[0]
    record["best_one"] = best_individual
    record["fitness_best_one"] = best_individual.fitness
    logbook.record(Generation=gen, evals=len(invalid_ind), **record)
    print(logbook.stream)



def exportCsv(csv_file_name, logbook):
    csv_columns = logbook[0].keys()
    csv_path = os.path.join(BASE_DIR, "results", csv_file_name)
    try:
        with open(csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in logbook:
                writer.writerow(data)
    except IOError:
        print("I/O error")


class nsgaAlgo(object):

    def __init__(self):
        self.json_instance = load_instance('./data/json/Input_Data.json')
        self.ind_size = self.json_instance['Number_of_customers']
        self.pop_size = 400
        self.cross_prob = 0.85
        self.mut_prob = 0.02
        self.num_gen = 150
        self.toolbox = base.Toolbox()
        self.logbook, self.stats = createStatsObjs()
        self.createCreators()

    def createCreators(self):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        self.toolbox.register('indexes', random.sample, range(1, self.ind_size + 1), self.ind_size)

        self.toolbox.register('individual', tools.initIterate, creator.Individual, self.toolbox.indexes)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register('evaluate', eval_indvidual_fitness, instance=self.json_instance, unit_cost=1)

        self.toolbox.register("select", tools.selNSGA2)

        self.toolbox.register("mate", cxOrderedVrp)

        self.toolbox.register("mutate", mutationShuffle, indpb=self.mut_prob)


    def generatingPopFitness(self):
        self.pop = self.toolbox.population(n=self.pop_size)
        self.invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self.fitnesses = list(map(self.toolbox.evaluate, self.invalid_ind))

        for ind, fit in zip(self.invalid_ind, self.fitnesses):
            ind.fitness.values = fit

        self.pop = self.toolbox.select(self.pop, len(self.pop))

        recordStat(self.invalid_ind, self.logbook, self.pop, self.stats, gen = 0)


    def runGenerations(self):
        for gen in range(self.num_gen):
            print(f"{20*'#'} Currently Evaluating {gen} Generation {20*'#'}")

            self.offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            self.offspring = [self.toolbox.clone(ind) for ind in self.offspring]

            for ind1, ind2 in zip(self.offspring[::2], self.offspring[1::2]):
                if random.random() <= self.cross_prob:
                    self.toolbox.mate(ind1, ind2)

                    del ind1.fitness.values, ind2.fitness.values
                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)

            self.invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]
            self.fitnesses = self.toolbox.map(self.toolbox.evaluate, self.invalid_ind)
            for ind, fit in zip(self.invalid_ind, self.fitnesses):
                ind.fitness.values = fit

            self.pop = self.toolbox.select(self.pop + self.offspring, self.pop_size)

            # Recording stats in this generation
            recordStat(self.invalid_ind, self.logbook, self.pop, self.stats, gen + 1)

        print(f"{20 * '#'} End of Generations {20 * '#'} ")


    def getBestInd(self):
        self.best_individual = tools.selBest(self.pop, 1)[0]

        print(f"Best individual is {self.best_individual}")
        print(f"Number of vechicles required are "
              f"{self.best_individual.fitness.values[0]}")
        print(f"Cost required for the transportation is "
              f"{self.best_individual.fitness.values[1]}")

        printRoute(routeToSubroute(self.best_individual, self.json_instance))

    def doExport(self):
        csv_file_name = f"{self.json_instance['instance_name']}_" \
                        f"pop{self.pop_size}_crossProb{self.cross_prob}" \
                        f"_mutProb{self.mut_prob}_numGen{self.num_gen}.csv"
        exportCsv(csv_file_name, self.logbook)

    def runMain(self):
        self.generatingPopFitness()
        self.runGenerations()
        self.getBestInd()
        self.doExport()



def nsga2vrp():

    json_instance = load_instance('./data/json/Input_Data.json')
    
    ind_size = json_instance['Number_of_customers']

    pop_size = 400
    cross_prob = 0.2
    mut_prob = 0.02
    num_gen = 220

    creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('indexes', random.sample, range(1,ind_size+1), ind_size)

    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    toolbox.register('evaluate', eval_indvidual_fitness, instance=json_instance, unit_cost = 1)

    toolbox.register("select", tools.selNSGA2)

    toolbox.register("mate", cxOrderedVrp)

    toolbox.register("mutate", mutationShuffle, indpb = mut_prob)

    logbook, stats = createStatsObjs()

    print(f"Generating population with size of {pop_size}")
    pop = toolbox.population(n=pop_size)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    fitnesses = list(map(toolbox.evaluate, invalid_ind))

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))

    print("Recording the Data and Statistics")
    recordStat(invalid_ind, logbook, pop, stats,gen=0)

    for gen in range(num_gen):
        print(f"######## Currently Evaluating {gen} Generation ######## ")

        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cross_prob:

                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values                 
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
   
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = toolbox.select(pop + offspring, pop_size)

        recordStat(invalid_ind, logbook, pop, stats,gen+1)

    print(f"{20*'#'} End of Generations {20*'#'} ")

    best_individual = tools.selBest(pop, 1)[0]

    print(f"Best individual is {best_individual}")
    print(f"Number of vechicles required are {best_individual.fitness.values[0]}")
    print(f"Cost required for the transportation is {best_individual.fitness.values[1]}")

    printRoute(routeToSubroute(best_individual, json_instance))

    print(f"Testing whether we can export logbook")
    print(logbook[1].keys())

    csv_file_name = f"{json_instance['instance_name']}_pop{pop_size}_crossProb{cross_prob}_mutProb{mut_prob}_numGen{num_gen}.csv"
    exportCsv(csv_file_name, logbook)


if __name__ == "__main__":
    print("Running file directly, Executing nsga2vrp")

    someinstance = nsgaAlgo()
    someinstance.runMain()


def testcosts():
    test_instance = load_instance('./data/json/Input_Data.json')

    sample_individual = [19, 5, 24, 7, 16, 23, 22, 2, 12, 8, 20, 25, 21, 18,11,15, 1, 14, 17, 6, 4, 13, 10, 3, 9]

    sample_ind_2 = random.sample(sample_individual, len(sample_individual))
    print(f"Sample individual is {sample_individual}")
    print(f"Sample individual 2 is {sample_ind_2}")

    print(f"Sample individual cost is {getRouteCost(sample_individual, test_instance, 1)}")
    print(f"Sample individual 2 cost is {getRouteCost(sample_ind_2, test_instance, 1)}")

    print(f"Sample individual fitness is {eval_indvidual_fitness(sample_individual, test_instance, 1)}")
    print(f"Sample individual 2 fitness is {eval_indvidual_fitness(sample_ind_2, test_instance, 1)}")

def testroutes():
    test_instance = load_instance('./data/json/Input_Data.json')

    sample_individual = [19, 5, 24, 7, 16, 23, 22, 2, 12, 8, 20, 25, 21, 18,11,15, 1, 14, 17, 6, 4, 13, 10, 3, 9]
    best_ind_300_gen = [16, 14, 12, 10, 15, 17, 21, 23, 11, 9, 8, 20, 18, 19, 13, 22, 25, 24, 5, 3, 4, 6, 7, 1, 2]


    sample_ind_2 = random.sample(sample_individual, len(sample_individual))
    print(f"Sample individual is {sample_individual}")
    print(f"Sample individual 2 is {sample_ind_2}")
    print(f"Best individual 300 generations is {best_ind_300_gen}")

    print(f"Subroutes for first sample individual is {routeToSubroute(sample_individual, test_instance)}")
    print(f"Subroutes for second sample indivudal is {routeToSubroute(sample_ind_2, test_instance)}")
    print(f"Subroutes for best sample indivudal is {routeToSubroute(best_ind_300_gen, test_instance)}")

    print(f"Vehicles for sample individual {getNumVehiclesRequired(sample_individual, test_instance)}")
    print(f"Vehicles for second sample individual {getNumVehiclesRequired(sample_ind_2, test_instance)}")
    print(f"Vehicles for best sample individual {getNumVehiclesRequired(best_ind_300_gen, test_instance)}")

def testcrossover():
    ind1 = [3,2,5,1,6,9,8,7,4]
    ind2 = [7,3,6,1,9,2,4,5,8]
    anotherind1 = [16, 14, 12, 7, 4, 2, 1, 13, 15, 8, 9, 6, 3, 5, 17, 18, 19, 11, 10, 21, 22, 23, 25, 24, 20]
    anotherind2 = [21, 22, 23, 25,16, 14, 12, 7, 4, 2, 1, 13, 15, 8, 9, 6, 3, 5, 17, 18, 19, 11, 10, 24, 20]


    newind7, newind8 = cxOrderedVrp(ind1, ind2)
    newind9, newind10 = cxOrderedVrp(anotherind1, anotherind2)

    print(f"InpInd1 is {ind1}")
    print(f"InpInd2 is {ind2}")
    print(f"newind7 is {newind7}")
    print(f"newind8 is {newind8}")
    print(f"newind9 is {newind9}")
    print(f"newind10 is {newind10}")

def testmutation():
    ind1 = [3,2,5,1,6,9,8,7,4]
    mut1 = mutationShuffle(ind1)

    print(f"Given individual is {ind1}")
    print(f"Mutation from first method {mut1}")