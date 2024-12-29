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
NETWORK_FILENAME = "onnx_models/model_10_76.onnx"
NUM_OF_INTERVALS = 10 # Should match the number in the input_file
TMOUT = 18000


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
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=3600)
    # This changes when we use a different network
    input_array = np.array([[5.488077417775099e-05, 0.0]]) # What is the output value of this case??
    # input_array = np.array([[0.765, 0.09709340072908054]])
    output_value = network.evaluate(input_array)
    # Output value should be -0.03259936
    print(output_value[0][0])

def init_intervals_basic_model(num_of_intervals=2):
    """
    Initialize a list of intervals for the basic model.

    Args:
        num_of_intervals (int): The number of intervals to initialize.

    Returns:
        list[Interval]: A list of Interval objects with specified bounds.
    """
    intervals = []
    for i in range(num_of_intervals):
        intervals.append(Interval(i, 0.0, 1.0))
    return intervals


def define_input_variables_bounds(intervals, network, inputVars):
    for i in range(len(intervals)):
        network.setLowerBound(inputVars[i], intervals[i].lower_bound)
        network.setUpperBound(inputVars[i], intervals[i].upper_bound)
    return

def define_output_constraints_for_property(network, outputVars):
    # Negated property to validate : y0 <= 0

    # The first form
    # in_eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
    # in_eq.addAddend(1, outputVars[0])
    #in_eq.setScalar(0.0)
    # network.addEquation(in_eq)

    # The second form
    network.setUpperBound(outputVars[0], -EPSILON)


def run_query(intervals: list[Interval], network_filename: str, timeout: int = 3600) -> IntervalResult:
    """
    Check for violations of the negated property (y0 <= -EPSILON)
    If this returns SAT - we found a violation of the original property (y0 >= 0)
    If this returns UNSAT - the original property holds
    """
    network = Marabou.read_onnx(network_filename)
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout)

    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    define_input_variables_bounds(intervals, network, inputVars)

    # Check negation of original property (y0 >= 0)
    # i.e., check if y0 <= -EPSILON is satisfiable
    network.setUpperBound(outputVars[0], -EPSILON)

    exit_code, vals, stats = network.solve(options=options)

    if exit_code == 'sat':
        try:
            assert vals[outputVars[0]] < 0
            # logging.info(f"Query solver input values: {vals[inputVars[0]]}, {vals[inputVars[1]]}, output value: {vals[outputVars[0]]} ")
        except AssertionError:
            logging.info(f"Input values: {vals[inputVars[0]]}, {vals[inputVars[1]]}")
            logging.info(f"Assertion failed: {vals[outputVars[0]]} >= 0")
            raise

    return exit_code, vals


def create_left_and_right_intervals(index_to_split, intervals):
    middle = (intervals[index_to_split].lower_bound + intervals[index_to_split].upper_bound) / 2
    next_index = (index_to_split + 1) % len(intervals)

    # Create copies of intervals
    intervals_left = []
    intervals_right = []

    for interval in intervals:
        intervals_left.append(interval.copy())
        intervals_right.append(interval.copy())

    current_depth = intervals[index_to_split].depth

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
    intervals = init_intervals_basic_model(5)
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
        intervals = init_intervals_basic_model(NUM_OF_INTERVALS)

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
    logging.debug(f"\nProcessing at depth {current_depth}:")
    logging.debug(f"Interval bounds: {[(i.lower_bound, i.upper_bound) for i in intervals]}")
    logging.debug(f"Volume: {interval_volume}")

    # Stop conditions based on depth and volume
    if current_depth >= max_depth or interval_volume < EPSILON:
        logging.debug(f"Stopping due to max depth or small volume.")

        # Check both conditions even at max_depth
        exit_code, vals = run_query(intervals, network_filename, query_timeout)
        logging.debug(f"run_query Exit code for interval at max depth or small volume: {exit_code}")

        if exit_code == 'TIMEOUT':
            logging.warning(f"Timeout in first query at depth {current_depth}")
            return 'TIMEOUT'
        elif exit_code == 'unsat':
            num_of_unsat[0] += interval_volume
            logging.debug(f"Max depth/small volume UNSAT, adding volume: {interval_volume}")
        elif exit_code == 'sat':
            # Found a violation, check if region is pure SAT
            exit_code_neg, vals = check_negation(intervals, network_filename, query_timeout)
            logging.debug(f"Negation check exit code: {exit_code_neg}")

            if exit_code_neg == 'TIMEOUT':
                logging.warning(f"Timeout in negation query at depth {current_depth}")
                return 'TIMEOUT'
            elif exit_code_neg == 'unsat':
                num_of_sat[0] += interval_volume
                logging.debug(f"Max depth/small volume pure SAT, adding volume: {interval_volume}")
            elif exit_code_neg == 'sat':
                # Mixed region at max depth - split volume equally
                num_of_sat[0] += interval_volume / 2
                num_of_unsat[0] += interval_volume / 2
                logging.debug(f"Max depth/small volume mixed region, splitting volume: {interval_volume}")
            else:
                return 'UNKNOWN'
        else:
            return 'UNKNOWN'

        current_violation_rate = calculate_current_violation_rate(num_of_sat, num_of_unsat)
        if current_depth % 4 == 0:
            logging.info(f"Current violation rate: {current_violation_rate:.3f}%")
        return 'SUCCESS'

    # Regular depth processing
    exit_code, vals = run_query(intervals, network_filename, query_timeout)
    logging.debug(f"First check (y0 <= -EPSILON): {exit_code}")

    if exit_code == 'TIMEOUT':
        logging.warning(f"Timeout in main query at depth {current_depth}")
        return 'TIMEOUT'
    elif exit_code == 'unsat':
        num_of_unsat[0] += interval_volume
        logging.debug(f"Found pure UNSAT region, volume: {interval_volume}")
        if current_depth % 4 == 0:
            current_vr = calculate_current_violation_rate(num_of_sat, num_of_unsat)
            logging.info(f"Current violation rate: {current_vr:.3f}%")
        return 'SUCCESS'
    elif exit_code == 'sat':
        # Mixed or SAT region - further analysis required
        logging.debug(f"Found SAT region or mixed region.")
    else:
        logging.warning(f"Unknown exit_code: {exit_code}")
        return 'UNKNOWN'

    # Check negation
    exit_code_neg, vals = check_negation(intervals, network_filename, query_timeout)
    logging.debug(f"Second check (y0 >= 0): {exit_code_neg}")

    if exit_code_neg == 'TIMEOUT':
        logging.warning(f"Timeout in negation at depth {current_depth}")
        return 'TIMEOUT'
    elif exit_code_neg == 'unsat':
        num_of_sat[0] += interval_volume
        logging.debug(f"Found pure SAT region, volume: {interval_volume}")
        if current_depth % 4 == 0:
            current_vr = (num_of_sat[0] / (num_of_sat[0] + num_of_unsat[0])) * 100 if (num_of_sat[0] +
                                                                                                   num_of_unsat[
                                                                                                       0]) > 0 else 0.0
            logging.info(f"Current violation rate: {current_vr:.3f}%")
        return 'SUCCESS'
    elif exit_code_neg == 'sat':
        # Process mixed region
        logging.debug("Found mixed region, splitting...")
        intervals_left, intervals_right, next_index = create_left_and_right_intervals(index_to_split, intervals)
        vol_before = num_of_sat[0] + num_of_unsat[0]

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

        # Volume conservation check
        total_volume = num_of_sat[0] + num_of_unsat[0]
        vol_added = total_volume - vol_before
        if abs(vol_added - interval_volume) > EPSILON:
            logging.error(f"Volume not conserved! Added {vol_added} but expected {interval_volume}")

        if current_depth % 4 == 0:
            current_violation_rate = calculate_current_violation_rate(num_of_sat, num_of_unsat)
            logging.info(f"Current violation rate: {current_violation_rate:.3f}%")
    else:
        logging.warning(f"Unknown negation exit code: {exit_code_neg}")
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

    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0][0]

    define_input_variables_bounds(intervals, network, inputVars)

    # Check original property: y0 >= 0
    network.setLowerBound(outputVars[0], 0)

    exit_code, vals, stats = network.solve(options=options)

    if exit_code == 'sat':
        try:
            assert vals[outputVars[0]] >= 0
            #logging.info(f"Negation Query solver input values: {vals[inputVars[0]]}, {vals[inputVars[1]]}, output value: {vals[outputVars[0]]} ")
        except AssertionError:
            logging.info(f"Negation error - Input values: {vals[inputVars[0]]}, {vals[inputVars[1]]}")
            logging.info(f"Negation error - Assertion failed: {vals[outputVars[0]]} < 0")
            raise

    return exit_code, vals

def parameter_sweep():
    network_filename = NETWORK_FILENAME
    # epsilons = [1e-12, 1e-13, 1e-14, 1e-15]
    # depths = [12, 16, 20]

    epsilons = [1e-6, 1e-7]
    depths = [4,8]

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



