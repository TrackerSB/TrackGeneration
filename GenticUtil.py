import numpy as np
import random
import math
import matplotlib.pyplot as plt

from TrackGenUtil import *

import copy


def crossover(a, b, param=-1):
    ab = []

    for i in range(0, len(a)):
        weight = random.uniform(0, 1)
        ab.append(weight * a[i] + (1 - weight) * b[i])

    #n = len(a)
    #rnd = 0

    #if(param >= 0): rnd = param
    #else: rnd = int(random.uniform(1, n))

    #ab = np.concatenate((a[0:rnd], b[rnd:n]))

    return ab


def mutate(a, mutationProb, mutationDeviation):
    for i in range(0, len(a)):
        if random.uniform(0, 1) < mutationProb:
            a[i] = a[i] + np.random.uniform(0, mutationDeviation)


def produceIndividual(length, maxAcc):
    indiv = []

    for j in range(0, length):
        dir = random.uniform(-math.pi, math.pi)
        abs = random.uniform(0, maxAcc)
        indiv.append(abs * np.array([math.cos(dir), math.sin(dir)]))

    return np.array(indiv)

def initPop(count, length, maxAcc, tries):
    pop = []

    for i in range(0, count):       
        indiv = produceIndividual(length, maxAcc)
        pop.append([indiv, fitness2(indiv)])
        
    for i in range(0, tries):
        pop.sort(key=lambda pair: pair[1])
        indiv = produceIndividual(length, maxAcc)
        f = fitness2(indiv)        
        if(f > pop[0][1]): pop[0] = [indiv, f]

    return [pair[0] for pair in pop]


def getProfile(accs):
    s = []
    v = []
    a = []

    accs = np.vstack((accs, completeAcc(accs)))
    p = polysFromAccs(accs)

    for q in p:
        s.append(q.startPos())
        v.append(q.startVel())
        a.append(q.startAcc())

    return s, v, a


def getAccRel(v, a):
    arel = []

    for i in range(0, len(v)):
        arel.append(rotate(a[i], v[i], rev=True))

    return arel


def getSteerAng(v, arel, carLen, errVal):
    steerAng = []

    for i in range(0, len(arel)):
        velabs = np.linalg.norm(v[i])
        steerSin = (arel[i][1] * carLen) / (2 * velabs * velabs)

        if steerSin <= 1 and steerSin >= -1:
            steerAng.append(math.asin(steerSin))
        else:
            steerAng.append(errVal)

    return steerAng


def getBuckets(accs):
    carLen = 1
    maxAng = math.pi / 3
    maxAcc = 1
    maxAccAng = 5
    bucketSize = 32

    s, v, a = getProfile(accs)

    vAbs = list(map(np.linalg.norm, v))
    aRel = np.array(getAccRel(v, a))
    aEff = aRel.transpose()[0]
    aAng = aRel.transpose()[1]
    steerAng = getSteerAng(v, aRel, carLen, -10)

    accBucket = [0] * bucketSize
    angBucket = [0] * bucketSize

    pointCount = 0

    for i in range(0, len(v)):
        accIdx = int((aEff[i] + maxAcc) * bucketSize / (2 * maxAcc))
        angIdx = int((steerAng[i] + maxAng) * bucketSize / (2 * maxAng))

        if accIdx >= 0 and accIdx < bucketSize: accBucket[accIdx] += 1
        if angIdx >= 0 and angIdx < bucketSize: angBucket[angIdx] += 1

        pointCount += 1

    return accBucket, angBucket


def fitness2(accs):
    accBucket, angBucket = getBuckets(accs)
    entrop = 0

    for i in range(0, len(angBucket)):
        if accBucket[i] > 0:
            accProb = accBucket[i] / len(accs)
            accEnt = -accProb * math.log(accProb, 2)
            entrop += accEnt

        if angBucket[i] > 0:
            angProb = angBucket[i] / len(accs)
            angEnt = -angProb * math.log(angProb, 2)
            entrop += angEnt

    return entrop


    #sin(steerang) * v * v <= awinmax

    #vAbs = list(map(lambda v: 1 / v, vAbs))
    #velang = np.array([vAbs, steerAng]).transpose()
    #velang = list(map(lambda x: rotate(x, [1, 1]), velang))
    #velangT = np.array(velang).transpose()
    #plt.figure(2)
    #plt.scatter(velangT[0], velangT[1])

    #plt.axis('equal')
    #plt.show()
def elminateDuplicates(pop, threshold):
    newPop = []

    for i in range(0, len(pop)):
        existsDup = False
        j = i + 1

        while (not existsDup) & (j < len(pop)):
            existsDup |= isDuplicate(pop[i], pop[j], threshold)
            j += 1

        if not existsDup:
            newPop.append(pop[i])

    return newPop


def elminateIntersecting(pop, delta):
    newPop = []

    for ind in pop:
        if not doBezIntersect(bezFromPolys(polysFromAccs(np.vstack((ind, completeAcc(ind))))), delta):
            newPop.append(ind)

    return newPop


def evolve(pop, times, relNewSize, duplicatesThreshold,
           intersectionDelta, mutationProb, mutationDeviation, tries):

   

    for i in range(0, times):
        pop = geneticStep(pop, relNewSize, duplicatesThreshold,
                          intersectionDelta, mutationProb,
                          mutationDeviation, tries) #(i % 10) == 0)

        print(f"STEP {i}")

    return pop


def crossoverUntilSize(pop, size):
    newPop = []

    if len(pop) == 1:
        #todo: handle
        pass

    for i in range(0, size):
        rnd1 = int(random.uniform(0, len(pop)))
        rnd2 = rnd1

        while (rnd2 == rnd1):
            rnd2 = int(random.uniform(0, len(pop)))

        newInd = crossover(pop[rnd1], pop[rnd2])
        newPop.append(newInd)

    return newPop

def geneticStep(pop, relNewSize, duplicatesThreshold,
                intersectionDelta, mutationProb,
                mutationDeviation, tries):

    initSize = len(pop)

    pop = pop + initPop(int(len(pop) / 10), len(pop[0]), 1, 0)

    newPop = []

    for i in range(0, initSize):   
        indiv = []
        for j in range(0, 100):
            idx1 = int(random.uniform(0, len(pop)))
            idx2 = int(random.uniform(0, len(pop)))
            indiv = crossover(pop[idx1], pop[idx2])
            mutate(indiv, mutationProb, mutationDeviation)
            if(not containsDuplicate(indiv, [indiv[0] for indiv in newPop], duplicatesThreshold)):
                break           
        newPop.append([indiv, fitness2(indiv)])        
        
    for i in range(0, tries):
        newPop.sort(key=lambda pair: pair[1])
        idx1 = int(random.uniform(0, len(pop)))
        idx2 = int(random.uniform(0, len(pop)))
        indiv = crossover(pop[idx1], pop[idx2])
        mutate(indiv, mutationProb, mutationDeviation)
        f = fitness2(indiv)                
        isDup = False

        if(f > newPop[0][1] and not containsDuplicate(indiv, [indiv[0] for indiv in newPop], duplicatesThreshold)):
            newPop[0] = [indiv, f]
            
          

    return [indiv[0] for indiv in newPop]
