import getopt

from common.geometry import Point


def parse(filename, args):
    arguments = {
        "gridMinimum": Point(0.0, 0.0),
        "gridMaximum": Point(20.0, 20.0),
        "vehicleFirst": Point(2.0, 2.0),
        "vehicleFinal": Point(18.0, 18.0),
        "objectSize": Point(1.0, 1.0),
        "obstacleCount": 40,
        "obstacleTheta": 0.0
    }
    if filename == "genetic-algorithm":
        arguments["populationCount"] = 80
        arguments["interpolation"] = 8
        arguments["pathSegments"] = 2
        arguments["curveSamples"] = 16
        arguments["mutationChance"] = 0.04
        arguments["evolutionMax"] = 10
    return arguments
