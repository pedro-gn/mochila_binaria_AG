import numpy as np
from readFiles import readFiles
import statistics
import matplotlib.pyplot as plt
def get_items_weight(samples, items_weight):
   
    return [np.sum(np.multiply(sample, items_weight)) for sample in samples]



def get_items_profits(samples, items_profits):

    return [np.sum(np.multiply(sample, items_profits)) for sample in samples]


def samples_avaliation(samples, items_profits, items_weight, knackpack_weight) :
    weights = get_items_weight(samples, items_weight)
    profits = get_items_profits(samples, items_profits)
    fitness = np.empty(len(samples))

    for i, w in enumerate(weights):

        if w <= knackpack_weight :
            fitness[i] = profits[i]
        else:
            fitness[i] = (profits[i] - (profits[i] * (np.sum(items_weight) - knackpack_weight)))

    return fitness


def samples_avaliation_normalized(samples, items_profits, items_weight, knackpack_weight) :
    weights = get_items_weight(samples, items_weight)
    profits = get_items_profits(samples, items_profits)
    fitness = np.empty(len(samples))

    for i, w in enumerate(weights):

        if w <= knackpack_weight :
            fitness[i] = profits[i]
        else:
            fitness[i] = (profits[i] - (profits[i] * (np.sum(items_weight) - knackpack_weight)))

    return [ f/np.sum(fitness) for f in fitness]



def tournament_selection(pop, pop_fit, k):
    begin = 0
    best = None
    best_fit = None
    for i in range(k):
        ind_number = np.random.randint(0,len(pop))
        
        ind = pop[ind_number] #individual
        ind_fit = pop_fit[ind_number] #individual fitness
        
        if begin == 0 or ind_fit > best_fit :
            begin = 1
            best = ind
            best_fit = ind_fit
    return best
    

def mutation(child1, child2, mutation_rate): 
    for i in range( len(child1) ):
        rand1 = np.random.random()
        rand2 = np.random.random()

        if rand1 <= (mutation_rate/100):
            child1[i] = int(not child1[i])

        if rand2 <= (mutation_rate/100):
            child1[i] = int(not child2[i])

    return (child1, child2)


def cross_breed(parent1, parent2):
    rand = np.random.randint(0,9)

    p1temp1 = parent1[:rand+1]
    p1temp2 = parent1[rand+1:]
    
    p2temp1 = parent2[:rand+1]
    p2temp2 = parent2[rand+1:]

    child1 = np.concatenate( (p1temp1, p2temp2) )
    
    child2 = np.concatenate( (p2temp1, p1temp2) )
    
    return (child1, child2)



def main(samples_number, mutation_rate, max_generations):

    knackpack_weight, items_profits, items_weight, solution  = readFiles()


    #first random population
    parents = np.random.randint(2, size=(samples_number,10))
    
   
    best_individuals_for_generation = []


    for i in range(max_generations): #generations
        
        parents_fit = samples_avaliation(parents, items_profits, items_weight, knackpack_weight)

        best_individuals_for_generation.append(int(np.max(parents_fit)))

        child = []

        for j in range( int(samples_number/2) - 1):
            #parents selection
            parent1 = tournament_selection(parents, parents_fit, 2)
            parent2 = tournament_selection(parents, parents_fit, 2)
            child1, child2 = cross_breed(parent1, parent2)
            child1, child2 = mutation(child1, child2, mutation_rate)

            child.append(child1)
            child.append(child2)


        #elitism gets 2 best parents
        elits = parents_fit.argsort()[-2:][::-1]
        for e in elits:
            child.append(parents[e])
            
        
        child = np.array(child) 
        parents = child

    return best_individuals_for_generation



if __name__ == "__main__" :
    mutatin_rates = (5, 10, 15)
    samples_numbers = (20, 50, 100)
    max_generations = (20, 50, 100)

    last = []
    means = []



    for mr in mutatin_rates :
        for sn in samples_numbers :
            for mg in max_generations :
                last = []
                generations = []
                for i in range(10):
                    x = main(sn, mr, mg)
                    
                    last.append(x[-1])


                means.append(statistics.mean(last))



                if statistics.mean(last) == 309 :
                    y = np.array(x)
                    x = np.array([n for n in range(mg)])

                    print(x)
                    print(y)
                    plt.plot(x, y)
                    plt.show()