import getopt

from common.geometry import Point


def parse(args):
    inputs = {
        "gridMinimum": Point(0.0, 0.0),
        "gridMaximum": Point(20.0, 20.0),
        "vehicleFirst": Point(2.0, 2.0),
        "vehicleFinal": Point(18.0, 18.0),
        "objectSize": Point(1.0, 1.0),
        "obstacleCount": 40,
        "obstacleTheta": 0.0
    }
    return inputs
