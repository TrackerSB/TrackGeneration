import numpy as np
import random
from time import sleep
import math
import matplotlib.pyplot as plt 
from beamngpy import BeamNGpy, Scenario, Road, Vehicle

from TrackGenUtil import *
from GenticUtil import *

def writeTrack(track, nr):
    file = open(f"track_{nr}.xml", "w")
    head = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?><environment xmlns=\"http://drivebuild.com\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://drivebuild.com ../drivebuild.xsd\"><author>Sebastian Asen</author><timeOfDay>0</timeOfDay><lanes><lane>\n"    
    tail = "</lane></lanes></environment>"

    file.write(head)

    for point in track:        
        file.write(f"<laneSegment x=\"{point[0]}\" y=\"{point[1]}\" width=\"5\"/>\n")     
        
    file.write(tail)    
    file.close()

def GenerateTrack(trackLength, sampleRate, show, startBeamNG):  
    populationSize = 100
    maxAcc = 1

    times = 20
    relNewSize = 0.6

    duplicatesThreshold = 0.05
    intersectionDelta = 0.01

    mutationProb = 0.1
    mutationDeviation = 0.01

    print("Generating Tracks")

    pop = initPop(populationSize, trackLength, maxAcc, 20)

    pop = evolve(pop, times, relNewSize, duplicatesThreshold,
           intersectionDelta, mutationProb, mutationDeviation, 10)

    print("eliminating intersecting tracks")
    pop = elminateIntersecting(pop, intersectionDelta)

    if show:
        plotTrack(pop, 100)

    tracks = []

    nr = 0

    for track in pop:
        track = np.vstack((track, completeAcc(track)))
        poly = polysFromAccs(track)
        bez = bezFromPolys(poly)
        smpl = sampleBezSeries(bez, sampleRate).transpose()
        smpl = scaleTrack(smpl, 100, 100)
        smpl = np.array(smpl)
        smpl = np.vstack((smpl, [smpl[0]]))
        tracks.append(smpl)
        writeTrack(smpl, nr)
        nr += 1


    if startBeamNG:   
        nodes = []
        for p in tracks[0]: nodes.append((p[0], p[1], 0, 7))

        beamng = BeamNGpy('localhost', 64256)
        vehicle = Vehicle('ego', model='etk800', licence='PYTHON', colour='Green')
        scenario = Scenario('GridMap_v0422', 'track test')
        scenario.add_vehicle(vehicle, pos=(0, 0, -16.64), rot=(0, 0, 180))
        road = Road(material='track_editor_C_center', rid='main_road', texture_length=5, looped=True)

        road.nodes.extend(nodes)
        scenario.add_road(road)
        scenario.make(beamng)

        beamng.open()
        beamng.load_scenario(scenario)
        beamng.start_scenario()
        vehicle.ai_set_mode('span')

        while 1:
            vehicle.update_vehicle()
            print(vehicle.state['pos'])
            sleep(1)
