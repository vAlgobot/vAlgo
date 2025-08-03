from utils.logger import get_logger
logger = get_logger(__name__)

R1 = 450
R2 = 490
R3 = 545
R4 = 585

PDH = 440
CP = 410
PDL = 380

S1 = 390
S2 = 350
S3 = 290
S4 = 230

LTP = 380

MAX_SL_POINTS = 50 #Maximum SL Point must be placed
SL1 = 30 #Breakout1 candle High-Low
SL2 = 40 #Breakout2 candle High-Low

support = True
resistance = False

def get_sl_points(sl1, sl2, max_sl_points):
    """
    Determines the stop loss (SL) points based on given SL1 and SL2 values.
    
    Args:
        sl1 (int): First stop loss value.
        sl2 (int): Second stop loss value.
        max_sl_points (int): Maximum stop loss points allowed.
    
    Returns:
        int: The selected stop loss value.
    """
    # logger.info(f"Calculating SL points: sl1={sl1}, sl2={sl2}, max_sl_points={max_sl_points}")
    sl = max_sl_points

    if sl1 <= max_sl_points and sl1 <= sl2:
        sl = sl1
    #if sl2 <= max_sl_points and sl2 <= sl1:
        #sl = sl2

    # logger.info(f"Calculated SL points: {sl}")
    return sl

def get_pivots_and_midpoints(pivots):
    """
    Generates a list of pivots and their midpoints.
    
    Args:
        pivots (list): List of pivot points.
    
    Returns:
        list: List containing pivots and their midpoints.
    """
    # logger.info(f"Calculating pivots and midpoints for {pivots}")
    pivots_and_midpoints = []

    for i in range(len(pivots) - 1):
        if not pivots_and_midpoints:
            pivots_and_midpoints.append(pivots[i])
        pivots_and_midpoints.append((pivots[i] + pivots[i + 1]) / 2)
        pivots_and_midpoints.append(pivots[i + 1])

    # logger.info(f"Generated pivots and midpoints: {pivots_and_midpoints}")
    return pivots_and_midpoints

def resistance_first_tp(pivots, ltp, sl, ratio):
    """
    Calculates the first take profit (TP) in a resistance zone.
    
    Args:
        pivots (list): List of pivot points.
        etp (int): Entry price.
        sl (int): Stop loss value.
        ratio (float): Risk-reward ratio.
    
    Returns:
        float: First take profit level.
    """
    # logger.info(f"Calculating resistance TP: etp={etp}, sl={sl}, ratio={ratio}")
    pivots = sorted(pivots)
    pivots_and_midpoints = get_pivots_and_midpoints(pivots)
    target_pivot_list = [item for item in pivots_and_midpoints if item >= ltp]
    # logger.info(f"Calculated target pivot list: {pivots_and_midpoints}")
    for item in target_pivot_list:
        tp = ltp + abs(item - ltp)
        min_tp = ltp + (sl * ratio)
        max_tp = ltp + (sl * 3.0)
        if tp <= max_tp and tp >= min_tp:
            # min_tp = min(sl * ratio, tp)
            # min_tp = min(float(sl) * float(ratio), float(tp))
            # first_tp = ltp - min_tp
            # logger.info(f"First TP found: {first_tp}")
            first_tp = tp
            return first_tp
    default_tp = ltp + (sl * ratio)
    # logger.info(f"No pivot match, defaulting TP to: {default_tp}")
    return default_tp

def support_first_tp(pivots, ltp, sl, ratio):
    """
    Calculates the first take profit (TP) in a support zone.
    
    Args:
        pivots (list): List of pivot points.
        etp (int): Entry price.
        sl (int): Stop loss value.
        ratio (float): Risk-reward ratio.
    
    Returns:
        float: First take profit level.
    """
    # logger.info(f"Calculating support TP: ltp={ltp}, sl={sl}, ratio={ratio}")
    pivots = sorted(pivots, reverse=True)
    pivots_and_midpoints = get_pivots_and_midpoints(pivots)
    target_pivot_list = [item for item in pivots_and_midpoints if item <= ltp]
    # logger.info(f"Calculated target pivot list: {pivots_and_midpoints}")
    for item in target_pivot_list:
        tp = ltp - abs(item - ltp)
        min_tp = ltp - (sl * ratio)
        max_tp = ltp - (sl * 3.0)
        if tp >= max_tp and tp <= min_tp:
            #min_tp = min(sl * ratio, tp)
            # min_tp = min(float(sl) * float(ratio), float(tp))
            #first_tp = ltp - min_tp
            # logger.info(f"First TP found: {first_tp}")
            first_tp = tp
            return first_tp
    default_tp = ltp - (sl * ratio)
    # logger.info(f"No pivot match, defaulting TP to: {default_tp}")
    return default_tp

def get_first_tp(zone_type, sl, ltp, ratio, pivots, min_pivot, max_pivot):
    """
    Determines the first take profit level based on the zone type.
    
    Args:
        zone_type (str): Type of zone ('resistance' or 'support').
        sl (int): Stop loss value.
        ltp (int): Entry price.
        ratio (float): Risk-reward ratio.
        pivots (list): List of pivot points.
        min_pivot (int): Minimum pivot level.
        max_pivot (int): Maximum pivot level.
    Returns:
        float: First take profit level.
    """
    # logger.info(f"Getting first TP for zone: {zone_type}, ltp={ltp}, sl={sl}")
    if zone_type == "Resistance":
        return resistance_first_tp(pivots, ltp, sl, ratio)
    if zone_type == "Support":
        return support_first_tp(pivots, ltp, sl, ratio)

def get_first_sl(zone_type, min_pivot, max_pivot, ltp, sl, ratio):
    """
    Calculates the first stop loss level.
    
    Args:
        zone_type (str): Type of zone ('resistance' or 'support').
        min_pivot (int): Minimum pivot level.
        max_pivot (int): Maximum pivot level.
        etp (int): Entry price.
        sl (int): Stop loss value.
        ratio (float): Risk-reward ratio.
    
    Returns:
        float: First stop loss level.
    """
    # logger.info(f"Calculating first SL for zone: {zone_type}, ltp={ltp}, sl={sl}")
    if zone_type == "Resistance":
        return ltp - sl
    if zone_type == "Support":
        return ltp + sl
    return sl * ratio

def ratio_part2(ratio):
    """
    Extracts the numeric part of the ratio string and converts it to float.
    
    Args:
        ratio (str): Ratio string in the format "1:X".
    
    Returns:
        float: Extracted numeric ratio value.
    """
    try:
        # logger.info(f"Risk-Reward ratio: {ratio}")
        return float(ratio.split(":")[-1])
    except Exception as e:
        logger.exception(f"Error converting ratio: {e}")
        return 2

def get_sl_tp(zone_type, ltp, pivots, sl1, sl2, max_sl_points, ratio):
    """
    Computes the first stop loss and take profit levels.
    
    Args:
        zone_type (str): Type of zone ('resistance' or 'support').
        Ltp (int): Entry price.
        pivots (list): List of pivot points.
        sl1 (int): First stop loss value.
        sl2 (int): Second stop loss value.
        max_sl_points (int): Maximum allowed stop loss points.
        ratio (str): Risk-reward ratio in "1:X" format.
    
    Returns:'
        tuple: First stop loss and first take profit values.
    """
    logger.info(f"Calculating SL and TP for zone: {zone_type}, ltp={ltp}, ratio={ratio}")
    sl_ratio = ratio_part2(ratio)
    sl = get_sl_points(sl1, sl2, max_sl_points)
    first_sl = round(get_first_sl(zone_type, min(pivots), max(pivots), ltp, sl, sl_ratio), 2)
    first_tp = round(get_first_tp(zone_type, sl, ltp, sl_ratio, pivots, min(pivots), max(pivots)), 2)
    logger.info(f"SL:{first_sl}, TP:{first_tp}")
    return first_sl, first_tp

# pivot_values = [S1,S2,S3,S4]
# get_sl_tp("Support", LTP, pivot_values, SL1, SL2, MAX_SL_POINTS, "1:2")


