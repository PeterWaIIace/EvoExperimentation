import numpy as np
import random

HALF = 8

def mixAdjacent(genomes,scores):
    genomes = sorted(zip(scores,genomes))

    new_genomes = []
    for n in range(0,len(genomes),2):
        new_genomes.append(crossover(genomes[n][1],genomes[n+1][1]))
        new_genomes.append(crossover(genomes[n+1][1],genomes[n][1]))

    return new_genomes

def mateRest(superior,rest):
    rest = rest
    if len(superior) > 0:
        for n,_ in enumerate(rest):
            randArray = np.random.randint(0,10,size=superior.shape)
            mask = randArray > HALF
            s_i = int((random.random()*len(superior)))
            rest[n][mask] = superior[s_i][mask]
    return rest

def mateSuperior(superior):
    if len(superior) > 0:
        for s in superior:
            randomInt = np.random.randint(0,10,size=superior.shape)
            mask = randomInt > HALF
            s_i_1 = int(random.random()*len(superior)-1)
            s_i_2 = int(random.random()*len(superior)-1)
            superior[s_i_1][mask] = superior[s_i_2][mask]
    return superior

def generateOffspring(parent1,parent2,population):
    offspring = []
    for n in range(population):
        genome = crossover(parent1,parent2)
        assert(genome.shape==parent1.shape)
        offspring.append(genome)

    return offspring

def separateSuperior(genomes,scores,cmp =lambda score,mean : score < mean):
    superior = []
    rest     = []

    sortedGenomes = [x for _, x in sorted(zip(scores, genomes), key=lambda pair: pair[0])]
    superior = sortedGenomes[:2]
    rest = sortedGenomes[2:]

    return (superior,rest)


def mixRandomly(genomes,scores,cmp=lambda score,mean : score <= mean):
    genomes = sorted(zip(scores,genomes), key=lambda x: x[0])

    superior, rest= separateSuperior(genomes,scores,cmp=cmp)

    ## TO DO: add at least one superior gen
    rest = mateRest(superior,rest)
    superior = mateSuperior(superior)

    new_genomes = rest + superior
    return new_genomes


def crossover(genome1, genome2):
    newGenome = genome2.copy()
    randArray = np.random.randint(0,10,size=genome1.shape)
    mask = randArray > HALF

    newGenome[mask] = genome1[mask]

    return newGenome


# NoM - number of mutations
def mutate(genome,NoM):
    genome_cp = genome.copy()
    genome = genome.copy()

    min_value = 0
    max_value = 4
    while(genome_cp == genome).all():
        randArray = np.random.randint(0,10,size=genome.shape)
        mask = randArray > HALF
        evolutionDrift = np.random.randint(-2,2,size=genome.shape)
        genome[mask] += evolutionDrift[mask]
        genome[(genome < min_value) | (genome > max_value)] %= max_value

    return genome

def __mutateMany(genomes,ms,rate=0.5):
    genomes = genomes.copy()
    if len(genomes):
        for n,genome in enumerate(genomes):
            if rate > random.random():
                genomes[n] = mutate(genome,ms)

    return genomes


forbiddenGenomes = dict()
# check if genomes are in forbidden dict
def filterGenomes(genomes):
    for n,genome in enumerate(genomes):
        while genome.tobytes() in forbiddenGenomes.keys():
            genome = mutate(genome,1)
        genomes[n] = genome

    return genomes

def addToForbiddenGenomes(genomes):
    for genome in genomes:
        forbiddenGenomes[genome.tobytes()] = 1

# msR - mutation strength of rest
# msS - mutation strength of superior
def mixAndMutate(genomes,scores,mr=0.5,ms=2,maxPopulation=50,genomePurifying=False):

    assert(isinstance(genomes[0],np.ndarray))

    superior, rest= separateSuperior(genomes,scores)
    copy_sup = superior.copy()

    # ## TO DO: add at least one superior gen
    # offspring = mateSuperior(superior)

    offspring = generateOffspring(superior[0],superior[1],len(genomes))
    newGeneration = __mutateMany(offspring,ms,rate=mr)
    # newGeneration = offspring 

    random.shuffle(newGeneration)
    newGeneration = newGeneration[:maxPopulation-1] + [copy_sup[0]]
    # newGeneration = newGeneration[:maxPopulation-2] + superior
    # newGeneration = newGeneration[:]

    if genomePurifying == True:
        addToForbiddenGenomes(rest)
        newGeneration = filterGenomes(newGeneration)

    assert(isinstance(newGeneration[0],np.ndarray))
    return newGeneration
