from pathlib import Path


def getTest():
    from TrackGeneration import GenerateTrack
    GenerateTrack(25, 3, False, False)
    return (Path("track_0.xml"), Path("criteria.xml"))
