""" 
This module was created to handle common
geo data and issues.

"""


from typing import Tuple, Union
from numpy import inf
import numpy as np
import geopy.distance


Lower = 'lower'
Upper = 'upper'
Latitude = 'lat'
Longitude = 'lon'


def distance_by_geodesic_coord(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Get distance in meters between two points by geodesic coordinates.

    Parameters
    ----------
    p1 : Tuple[float, float]
        Coordinates of point 1 in format (Latitude, Longitude).
    p2 : Tuple[float, float]
        Coordinates of point 2 in format (Latitude, Longitude).

    Returns
    -------
    float
        The distance in meters between two points by geodesic coordinates. 
    """
    return geopy.distance.geodesic(p1, p2).m


def find_y_value(
    *args, **kwargs
) -> any:
    interval_arr = args[0]
    geodesic_axis = args[1]
    if geodesic_axis == 'lat':
        y_bot_value = geopy.distance.geodesic(
            (0, 0), (interval_arr[0], 0)
        ).m
        y_top_value = geopy.distance.geodesic(
            (0, 0), (interval_arr[1], 0)
        ).m
    elif geodesic_axis == 'lon':
        y_bot_value = geopy.distance.geodesic(
            (0, 0), (0, interval_arr[0])
        ).m
        y_top_value = geopy.distance.geodesic(
            (0, 0), (0, interval_arr[1])
        ).m
    return y_bot_value, y_top_value


def distance_to_geodesic(
    distance: float,
    init_guess_interval: Tuple[float, float]=(0, 1),
    geodesic_axis: Union[Latitude, Longitude]=Latitude,
    precision: float=1,
    boundary: Union[Lower, Upper]=Lower
) -> Tuple[
    float, 
    float, 
    Tuple[float, float],
    int
]:
    """Tranform distance in meters to corresponding latitude or 
    longitude angle value.

    Parameters
    ----------
    distance : float
        Distance in meters
    init_guess_interval : Tuple[float, float], optional
        Guess interval where corresponding value will 
        be, by default (0, 1)
    geodesic_axis : Union[Latitude, Longitude], optional
        Value of angle on Latitude or Longitude axis, by default Latitude
    precision : float, optional
        Precision of angle value, by default 1
    boundary : Union[Lower, Upper], optional
        If Lower, returns value of angle corresponding
        to some distance smaller than **distance**. If 
        Upper, returns value of angle corresponding
        to some distance bigger than **distance**, 
        by default Lower

    Returns
    -------
    Tuple[ float, float, Tuple[float, float], int ]
        Returns the distance corresponding to desired angle, 
        the angle value, the interval containing the real angle value 
        and the number of iterations until find the value of angle
    """
    
    y_value = inf
    interval_arr = np.array(init_guess_interval)
    
    boundary_condition = lambda y, val: \
        y > val if boundary == Lower else \
        y < val if boundary == Upper else \
        False
    
    iters = 0
    while abs(y_value - distance) > precision or boundary_condition(y_value, distance):
        
        # find y value
        y_bot_value, y_top_value = find_y_value(
            interval_arr, geodesic_axis
        )
        
        # verify inequality between value and y value
        if y_bot_value <= distance <= y_top_value:
            interval_mean = interval_arr.mean()
            y_mean_value, _ = find_y_value(
                np.array([interval_mean, interval_mean]), geodesic_axis
            )
            if y_mean_value > distance:
                interval_arr = np.array([
                    interval_arr[0], interval_mean
                ])
            elif y_mean_value < distance:
                interval_arr = np.array([
                    interval_mean, interval_arr[1]
                ])
            else:
                interval_arr = np.array([
                    interval_mean, interval_mean
                ])
        else:
            interval_arr = np.array([
                interval_arr[1],
                2*interval_arr[1] - interval_arr[0],
            ])
        
        y_value = y_mean_value
        iters = iters + 1
    
    return y_value, interval_mean, tuple(interval_arr), iters