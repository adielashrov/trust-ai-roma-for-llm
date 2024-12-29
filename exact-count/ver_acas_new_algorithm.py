import logging
import timeit
from datetime import datetime
import os
import time
import traceback
from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum

import numpy as np
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils
from sympy.plotting.intervalmath import interval
from sympy.polys.polyconfig import query

EPSILON = 0.00001
DISCRETIZATION_FACTOR = 0.01
MAX_DEPTH = 16
NETWORK_FILENAME = "acasxu/ACASXU_experimental_v2a_2_7.onnx"
NUM_OF_INTERVALS = 5 # unused in this case
TMOUT = 18000
NUMERIC_TOLERANCE  = 1e-6

class IntervalStatus(Enum):
    UNPROCESSED = "unprocessed"
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    NEEDS_SPLIT = "needs_split"


@dataclass
class IntervalResult:
    status: IntervalStatus
    exit_code: str = None


class Interval:
    def __init__(self, index, lower_bound, upper_bound, depth=0):
        self.index = index
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.depth = depth
        self.status = IntervalStatus.UNPROCESSED

    def __str__(self):
        return f"Index {self.index}: [{self.lower_bound}, {self.upper_bound}] (depth: {self.depth}, status: {self.status})"

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        return (self.index == other.index and
                self.lower_bound == other.lower_bound and
                self.upper_bound == other.upper_bound and
                self.depth == other.depth)

    def __repr__(self):
        return f"Interval(index={self.index}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, depth={self.depth})"

    def __hash__(self):
        return hash((self.index, self.lower_bound, self.upper_bound, self.depth))

    def copy(self):
        """Create a deep copy of the interval"""
        return Interval(
            self.index,
            self.lower_bound,
            self.upper_bound,
            self.depth
        )


def setup_logging(network_filename):
    model_name = os.path.splitext(os.path.basename(network_filename))[0]
    log_filename = f'verification_{model_name}_%d_%m_%H_%M_%S.log'
    log_filename = datetime.now().strftime(log_filename)
    logging.basicConfig(filename=log_filename, level=logging.DEBUG,  # Changed to DEBUG
                       format='%(asctime)s - %(levelname)s - %(message)s')
    return log_filename


def test_marabou_solution():
    network_filename = "onnx_models/model_5_09.onnx"
    network = Marabou.read_onnx(network_filename)
    options = Marabou.createOptions(verbosity=1, timeoutInSeconds=3600)
    # This changes when we use a different network
    input_array = np.array([[5.488077417775099e-05, 0.0]]) # What is the output value of this case??
    # input_array = np.array([[0.765, 0.09709340072908054]])
    output_value = network.evaluate(input_array)
    # Output value should be -0.03259936
    print(output_value[0][0])

def init_intervals_acas_model():
    """
    Initialize a list of intervals for the acas model.

    Args:
        num_of_intervals (int): The number of intervals to initialize.

    Returns:
        list[Interval]: A list of Interval objects with specified bounds.
    """
    intervals = []
    intervals.append(Interval(0, 0.6, 0.6798577687))
    intervals.append(Interval(1, -0.5, 0.5))
    intervals.append(Interval(2, -0.5, 0.5))
    intervals.append(Interval(3, 0.45, 0.5))
    intervals.append(Interval(4, -0.5, -0.45))
    return intervals


def define_input_variables_bounds(intervals, network, inputVars):
    for i in range(len(intervals)):
        network.setLowerBound(inputVars[i], intervals[i].lower_bound)
        network.setUpperBound(inputVars[i], intervals[i].upper_bound)
    return

def define_output_constraints_for_property(network, outputVars):
    network.addInequality(vars=[outputVars[1], outputVars[0]], coeffs=[1, -1], scalar=0.0) # +y1-y0 <= 0
    network.addInequality(vars=[outputVars[2], outputVars[0]], coeffs=[1, -1], scalar=0.0) # +y2-y0 <= 0
    network.addInequality(vars=[outputVars[3], outputVars[0]], coeffs=[1, -1], scalar=0.0) # +y3-y0 <= 0
    network.addInequality(vars=[outputVars[4], outputVars[0]], coeffs=[1, -1], scalar=0.0) # +y4-y0 <= 0


# This function returns True if y0 is the maximum value among y0, y1, y2, y3, y4
# Otherwise, there is a different maximum value among y0, y1, y2, y3, y4
# and the function returns False
def check_run_query_solution(vals, outputVars):
    """
    Check if y0 is maximal. Returns True if y0 is maximal or within tolerance.
    """

    y0_val = vals[outputVars[0]]
    differences = []

    for i in range(1, 5):
        difference = vals[outputVars[i]] - y0_val
        differences.append(difference)
        if difference > NUMERIC_TOLERANCE:  # Only fail if difference exceeds tolerance
            logging.warning(f"Property check - y{i} exceeds y0 beyond tolerance:")
            logging.warning(f"  y0 = {y0_val}")
            logging.warning(f"  y{i} = {vals[outputVars[i]]}")
            logging.warning(f"  difference = {difference}")
            return False

    # If we got here, all differences are either negative or within tolerance
    # if max(differences) > 0:
    #    logging.debug("Property check - small positive differences found but within tolerance:")
    #   logging.debug(f"  y0 = {y0_val}")
    #    logging.debug(f"  differences = {differences}")

    return True  # Return True for both negative differences and differences within tolerance

def run_query(intervals: list[Interval], network_filename: str, timeout: int = 3600) -> IntervalResult:
    network = Marabou.read_onnx(network_filename)
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout)

    inputVars = network.inputVars[0][0][0][0]
    outputVars = network.outputVars[0][0]

    define_input_variables_bounds(intervals, network, inputVars)
    define_output_constraints_for_property(network, outputVars)

    exit_code, vals, stats = network.solve(options=options)

    if exit_code == 'sat':
        if not check_run_query_solution(vals, outputVars):
            logging.debug("Numerical precision issue in run_query solution:")
            logging.debug(f"Input values: {[vals[inputVars[i]] for i in range(5)]}")
            logging.debug(f"Output values: {[vals[outputVars[i]] for i in range(5)]}")
            # Consider this as UNSAT since the property doesn't strictly hold
            return 'unsat', vals

    return exit_code, vals


# This function returns True if y0 is NOT the maximal value among y0, y1, y2, y3, y4
# Otherwise, y0 is the maximal value among y0, y1, y2, y3, y4 and return False
def check_negated_property(vals, outputVars):
    """
    Check if y0 is NOT the maximum value among the outputs.
    Returns True if any yi is greater than or equal to y0, considering the numeric tolerance.
    """

    y0_val = vals[outputVars[0]]
    differences = []

    for i in range(1, 5):
        difference = vals[outputVars[i]] - y0_val
        differences.append(difference)
        if difference >= -NUMERIC_TOLERANCE:  # Changed to >= -NUMERIC_TOLERANCE
            # logging.debug(f"Negated property check - y{i} greater or equal to y0 (within tolerance):")
            #logging.debug(f"  y0 = {y0_val}")
            #logging.debug(f"  y{i} = {vals[outputVars[i]]}")
            #logging.debug(f"  difference = {difference}")
            return True

    logging.warning("Negated property check - all outputs less than y0:")
    logging.warning(f"  y0 = {y0_val}")
    logging.warning(f"  differences = {differences}")

    return False

def define_output_constraints_for_negated_property(network, outputVars, epsilon=EPSILON):
    """
    Check if y0 is NOT maximal by checking if any other output is strictly greater
    The negation of 'y0 is maximal' is 'exists i where yi > y0'
    """
    # Create equations for yi - y0 > 0 for some i
    equations = []
    for i in range(1, 5):  # For y1 through y4
        equation = MarabouUtils.Equation(MarabouCore.Equation.GE)  # Greater than
        equation.addAddend(1, outputVars[i])  # +yi
        equation.addAddend(-1, outputVars[0])  # -y0
        equation.setScalar(epsilon)  # yi - y0 >= epsilon
        equations.append([equation])

    # Add disjunction: at least one output should be greater than y0
    network.addDisjunctionConstraint(equations)

def validate_volume_split(parent_volume, left_volume, right_volume, current_depth):
    """Validate that split volumes sum to parent volume"""
    if abs((left_volume + right_volume) - parent_volume) > EPSILON:
        logging.error(f"Volume validation failed at depth {current_depth}:")
        logging.error(f"Parent volume: {parent_volume}")
        logging.error(f"Left volume:   {left_volume}")
        logging.error(f"Right volume:  {right_volume}")
        logging.error(f"Sum:           {left_volume + right_volume}")
        logging.error(f"Difference:    {abs((left_volume + right_volume) - parent_volume)}")
        return False
    return True

def create_left_and_right_intervals(index_to_split, intervals):
    middle = (intervals[index_to_split].lower_bound + intervals[index_to_split].upper_bound) / 2
    next_index = (index_to_split + 1) % len(intervals)

    intervals_left = []
    intervals_right = []

    # Deep copy all intervals
    for interval in intervals:
        intervals_left.append(interval.copy())
        intervals_right.append(interval.copy())

    current_depth = intervals[index_to_split].depth

    # Create new intervals for split dimension
    intervals_left[index_to_split] = Interval(
        intervals[index_to_split].index,
        intervals[index_to_split].lower_bound,
        middle,
        depth=current_depth + 1
    )

    intervals_right[index_to_split] = Interval(
        intervals[index_to_split].index,
        middle,
        intervals[index_to_split].upper_bound,
        depth=current_depth + 1
    )

    # Validate split volume
    original_volume = calculate_volume(intervals)
    left_volume = calculate_volume(intervals_left)
    right_volume = calculate_volume(intervals_right)

    if not validate_volume_split(original_volume, left_volume, right_volume, current_depth):
        logging.error("Volume validation failed in create_left_and_right_intervals")

    return intervals_left, intervals_right, next_index



def run_experiment(network_filename):
    logging.info("Starting verification process")
    logging.info(f"Network file: {network_filename}")
    logging.info(f"Current epsilon value: {-EPSILON}")
    logging.info(f"Current DISCRETIZATION_FACTOR: {DISCRETIZATION_FACTOR}")
    logging.info(f"Max depth: {MAX_DEPTH}")

    num_of_sat = [0.0]
    num_of_unsat = [0.0]

    # Initialize 5D unit hypercube
    intervals = init_intervals_acas_model()
    logging.info(f"Initial intervals: {intervals}")

    # Verify initial volume is 1.0
    initial_volume = 1.0
    for interval in intervals:
        initial_volume *= (interval.upper_bound - interval.lower_bound)
    logging.info(f"Initial volume check: {initial_volume}")

    calculate_violation_rate(network_filename, intervals, num_of_sat, num_of_unsat,
                             index_to_split=0, max_depth=MAX_DEPTH, timeout = TMOUT)

    total_volume = num_of_sat[0] + num_of_unsat[0]

    # Add validation checks
    if abs(total_volume - 1.0) > EPSILON:
        logging.warning(f"Total volume {total_volume} differs significantly from 1.0")

    if total_volume > 0:
        violation_rate = (num_of_sat[0] / total_volume) * 100
    else:
        logging.error("Total volume is 0!")
        violation_rate = 0.0

    logging.info(f"SAT volume: {num_of_sat[0]:.10f}")
    logging.info(f"UNSAT volume: {num_of_unsat[0]:.10f}")
    logging.info(f"Total volume: {total_volume:.10f}")
    logging.info(f"Violation rate: {violation_rate:.3f}%")


def run_experiment_with_params(network_filename, epsilon, max_depth, query_timeout=3600):
    global EPSILON
    EPSILON = epsilon

    start_time = time.time()

    logging.info(f"\nTesting with EPSILON={epsilon}, MAX_DEPTH={max_depth}")
    logging.info("Starting verification process")
    logging.info(f"Network file: {network_filename}")
    logging.info(f"Current epsilon value: {epsilon}")
    logging.info(f"Current DISCRETIZATION_FACTOR: {DISCRETIZATION_FACTOR}")
    logging.info(f"Max depth: {max_depth}")

    try:
        num_of_sat = [0.0]
        num_of_unsat = [0.0]
        intervals = init_intervals_acas_model()

        status = calculate_violation_rate(network_filename, intervals, num_of_sat, num_of_unsat,
                                          index_to_split=0, max_depth=max_depth, query_timeout=query_timeout)

        if status == 'TIMEOUT':
            logging.warning(f"Calculation terminated due to timeout")
            return 'TIMEOUT', None, None
        elif status == 'UNKNOWN':
            logging.warning(f"Calculation terminated with unknown status")
            return 'UNKNOWN', None, None
        elif status != 'SUCCESS':
            logging.error(f"Unexpected status: {status}")
            return 'UNKNOWN', None, None

        total_volume = num_of_sat[0] + num_of_unsat[0]

        logging.info(f"\nVolume Analysis:")
        logging.info(f"SAT volume:   {num_of_sat[0]:.10f}")
        logging.info(f"UNSAT volume: {num_of_unsat[0]:.10f}")
        logging.info(f"Total volume: {total_volume:.10f}")
        logging.info(f"Volume error: {abs(1.0 - total_volume):.10f}")

        if abs(total_volume - 1.0) > EPSILON:
            logging.warning(f"Volume mismatch! Total volume: {total_volume}")

        violation_rate = (num_of_sat[0] / total_volume) * 100 if total_volume > 0 else 0.0
        execution_time = time.time() - start_time

        logging.info(f"Violation rate: {violation_rate:.3f}%")
        logging.info(f"Execution time: {execution_time:.2f} seconds")

        return 'SUCCESS', violation_rate, execution_time

    except Exception as e:
        logging.error(f"Unexpected error in experiment: {str(e)}")
        return 'UNKNOWN', None, None

def calculate_current_violation_rate(num_of_sat, num_of_unsat):
    if (num_of_sat[0] + num_of_unsat[0]) > 0:
        return (num_of_sat[0] / (num_of_sat[0] + num_of_unsat[0])) * 100
    else:
        return 0.0

def calculate_violation_rate(network_filename: str,
                             intervals: list[Interval],
                             num_of_sat: list[float],
                             num_of_unsat: list[float],
                             index_to_split=0,
                             max_depth=3,
                             query_timeout=3600):
    interval_volume = calculate_volume(intervals)
    current_depth = intervals[0].depth
    vol_before = num_of_sat[0] + num_of_unsat[0]

    logging.debug(f"\n{'=' * 50}")
    logging.debug(f"Processing depth {current_depth}, index {index_to_split}")
    logging.debug(f"Current intervals: {[(i.lower_bound, i.upper_bound) for i in intervals]}")
    logging.debug(f"Current volume: {interval_volume}")
    logging.debug(f"Current SAT total: {num_of_sat[0]}, UNSAT total: {num_of_unsat[0]}")

    # Stop conditions based on depth and volume
    if current_depth >= max_depth or interval_volume < EPSILON:
        logging.info(f"Reached stop condition - Depth: {current_depth}, Volume: {interval_volume}")

        # First query at max depth to determine if interval is purely UNSAT or SAT
        exit_code, vals = run_query(intervals, network_filename, query_timeout)
        logging.debug(f"First query at max depth: {exit_code}")

        if exit_code == 'TIMEOUT':
            logging.warning(f"Timeout in first query at depth {current_depth}")
            return 'TIMEOUT'
        elif exit_code == 'unsat':
            num_of_unsat[0] += interval_volume
            logging.info(f"Max depth pure UNSAT region found, adding volume: {interval_volume}")
        elif exit_code == 'sat':
            # Run negation check to verify if it's fully SAT
            exit_code_neg, vals = check_negation(intervals, network_filename, query_timeout)
            logging.debug(f"Negation query at max depth: {exit_code_neg}")

            if exit_code_neg == 'TIMEOUT':
                logging.warning(f"Timeout in negation query at depth {current_depth}")
                return 'TIMEOUT'
            elif exit_code_neg == 'unsat':
                num_of_sat[0] += interval_volume
                logging.info(f"Max depth pure SAT region found, adding volume: {interval_volume}")
            elif exit_code_neg == 'sat':
                # Mixed region at max depth
                logging.info(f"Max depth mixed region found, splitting volume {interval_volume} equally")
                num_of_sat[0] += interval_volume / 2
                num_of_unsat[0] += interval_volume / 2
                logging.debug(f"After mixed split at max depth - SAT: {num_of_sat[0]}, UNSAT: {num_of_unsat[0]}")
            else:
                logging.error(f"Unknown exit code during negation at depth {current_depth}")
                return 'UNKNOWN'
        else:
            logging.error(f"Unknown exit code at depth {current_depth}")
            return 'UNKNOWN'

        current_vr = calculate_current_violation_rate(num_of_sat, num_of_unsat)
        logging.info(f"End of max depth processing - Violation Rate (VR): {current_vr:.3f}%")
        return 'SUCCESS'

    # Regular depth processing - query if interval is SAT or UNSAT
    exit_code, vals = run_query(intervals, network_filename, query_timeout)
    logging.debug(f"Regular depth query result: {exit_code}")

    if exit_code == 'TIMEOUT':
        logging.warning(f"Timeout in main query at depth {current_depth}")
        return 'TIMEOUT'
    elif exit_code == 'unsat':
        num_of_unsat[0] += interval_volume
        logging.info(f"Pure UNSAT region found at depth {current_depth}, adding volume: {interval_volume}")
        return 'SUCCESS'
    elif exit_code == 'sat':
        # Run negation to check if it's fully SAT or mixed
        exit_code_neg, vals = check_negation(intervals, network_filename, query_timeout)
        logging.debug(f"Negation query result: {exit_code_neg}")

        if exit_code_neg == 'TIMEOUT':
            logging.warning(f"Timeout in negation at depth {current_depth}")
            return 'TIMEOUT'
        elif exit_code_neg == 'unsat':
            num_of_sat[0] += interval_volume
            logging.info(f"Pure SAT region found at depth {current_depth}, adding volume: {interval_volume}")
            return 'SUCCESS'
        elif exit_code_neg == 'sat':
            logging.info(f"Mixed region found at depth {current_depth}, splitting for further analysis")
            intervals_left, intervals_right, next_index = create_left_and_right_intervals(index_to_split, intervals)

            # Process left interval
            status_left = calculate_violation_rate(network_filename, intervals_left, num_of_sat, num_of_unsat,
                                                   next_index, max_depth, query_timeout)
            if status_left != 'SUCCESS':
                return status_left

            # Process right interval
            status_right = calculate_violation_rate(network_filename, intervals_right, num_of_sat, num_of_unsat,
                                                    next_index, max_depth, query_timeout)
            if status_right != 'SUCCESS':
                return status_right

            # After processing both sub-intervals, validate volume conservation
            vol_after = num_of_sat[0] + num_of_unsat[0]
            vol_added = vol_after - vol_before
            if abs(vol_added) > 0 and abs(vol_added - interval_volume) > EPSILON:
                logging.error(f"Volume conservation failed at depth {current_depth}:")
                logging.error(f"Expected to add: {interval_volume}")
                logging.error(f"Actually added:  {vol_added}")
                logging.error(f"Difference:      {abs(vol_added - interval_volume)}")
                return 'UNKNOWN'
        else:
            logging.error(f"Unknown exit code during negation check: {exit_code_neg}")
            return 'UNKNOWN'

    return 'SUCCESS'


def calculate_volume(intervals):
    """Helper function to calculate volume of an interval set"""
    volume = 1.0
    for interval in intervals:
        volume *= (interval.upper_bound - interval.lower_bound)
    return volume


def check_negation(intervals, network_filename, timeout=3600):
    """
    Check the original property (y0 >= 0)
    """
    network = Marabou.read_onnx(network_filename)
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout)

    inputVars = network.inputVars[0][0][0][0]
    outputVars = network.outputVars[0][0]

    define_input_variables_bounds(intervals, network, inputVars)
    define_output_constraints_for_negated_property(network, outputVars)

    exit_code, vals, stats = network.solve(options=options)

    if exit_code == 'sat':
        if not check_negated_property(vals, outputVars):
            logging.warning("Numerical precision issue in negation check:")
            logging.warning(f"Input values: {[vals[inputVars[i]] for i in range(5)]}")
            logging.warning(f"Output values: {[vals[outputVars[i]] for i in range(5)]}")
            # Consider this as UNSAT since the negated property doesn't strictly hold?
            return 'unsat', vals

    return exit_code, vals


def parameter_sweep():
    network_filename = NETWORK_FILENAME
    # epsilons = [1e-12, 1e-13, 1e-14, 1e-15]
    # depths = [12, 16, 20]

    epsilons = [1e-6, 1e-8, 1e-10, 1e-12]
    depths = [8,12,16,20]

    results = {}

    logging.info("\nParameter Sweep Results:")
    logging.info(f"{'EPSILON':<10} {'MAX_DEPTH':<10} {'VR (%)':<10} {'Time (s)':<10}")
    logging.info("-" * 45)

    for epsilon in epsilons:
        for depth in depths:
            try:
                status, rate, exec_time = run_experiment_with_params(network_filename, epsilon, depth,
                                                                     query_timeout=TMOUT)

                if status == 'TIMEOUT':
                    logging.warning(
                        f"Parameter sweep terminated due to timeout at EPSILON={epsilon}, MAX_DEPTH={depth}")
                    return
                elif status == 'UNKNOWN':
                    logging.warning(
                        f"Parameter sweep terminated with unknown status at EPSILON={epsilon}, MAX_DEPTH={depth}")
                    return
                elif status != 'SUCCESS':
                    logging.error(f"Unexpected status {status} at EPSILON={epsilon}, MAX_DEPTH={depth}")
                    return

                results[(epsilon, depth)] = (rate, exec_time)
                logging.info(f"Summary: {epsilon:<10.0e} {depth:<10d} {rate:<10.3f} {exec_time:<10.2f}")
            except Exception as e:
                logging.error(f"Error with EPSILON={epsilon}, MAX_DEPTH={depth}: {str(e)}")
                results[(epsilon, depth)] = ("Error", 0)
            logging.info("-" * 45)

    # Print summary sorted by closeness to 9%
    valid_results = {k: v for k, v in results.items() if isinstance(v[0], (int, float))}
    sorted_results = sorted(valid_results.items(), key=lambda x: abs(x[1][0] - 9.0))

    logging.info("\nResults sorted by closeness to 9%:")
    for (epsilon, depth), (rate, exec_time) in sorted_results:
        logging.info(f"EPSILON={epsilon}, MAX_DEPTH={depth}: VR={rate:.3f}%, Time={exec_time:.2f}s")


if __name__ == '__main__':
    log_filename = setup_logging(NETWORK_FILENAME)
    start_total = time.time()
    parameter_sweep()
    end_total = time.time()
    logging.info(f"\nTotal sweep time: {end_total - start_total:.2f} seconds")



