#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from arl.test_support import import_visibility_from_oskar, export_visibility_to_fits

# Parse arguments
parser = argparse.ArgumentParser(description='Generate kernel file for')
parser.add_argument('--out', dest='out', type=str, required=True,
                    help='output prefix')
parser.add_argument('--ff', dest='ff', type=int, required=True,
                    help='Far field size')
parser.add_argument('--wstep', dest='wstep', type=int, required=True,
                    help='Step length for w coordinates')
parser.add_argument('--wcount', dest='wcount', type=int, required=True,
                    help='Number of w-planes to generate kernels for')
parser.add_argument('--size', dest='size', type=int, required=True,
                    help='Size of kernels to generate')
args = parser.parse_args()

# Open file

