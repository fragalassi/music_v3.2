#!/usr/bin/python

import argparse
import animaMSExamPreparation as examPreparation

parser = argparse.ArgumentParser(
    prog='animaMSExamPreparation.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Registers and pre-processes input images of an MS patient sequence onto a common reference.")

parser.add_argument('-r', '--reference', required=True, help='Path to the MS patient reference image (usually FLAIR at first time point)')
parser.add_argument('-f', '--flair', required=True, help='Path to the MS patient FLAIR image to register')
parser.add_argument('-t', '--t1', required=True, help='Path to the MS patient T1 image to register')
parser.add_argument('-g', '--t1-gd', default="", help='Path to the MS patient T1-Gd image to register')
parser.add_argument('-T', '--t2', default="", help='Path to the MS patient T2 image to register')
parser.add_argument('-o','--outputFolder',required=True,help='path to output image')

args = parser.parse_args()


examPreparation.process(args.reference, args.flair, args.t1, args.t1_gd, args.t2, args.outputFolder)