#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaMusicLesionSegmentation.py ..." has to be run

import os
import argparse
import animaMusicLesionSegmentation_v3 as lesionSegmentation

parser = argparse.ArgumentParser(
    prog='animaMusicLesionSegmentation_v3.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Compute MS lesion segmentation using a cascaded CNN. Uses preprocessed images from animaMSExamRegistration.py')

parser.add_argument('-f', '--flair', required=True, help='Path to the MS patient FLAIR image')
parser.add_argument('-t', '--t1', required=True, help='Path to the MS patient T1 image')
parser.add_argument('-T', '--t2', required=False, help='Path to the MS patient T2 image')
parser.add_argument('-m', '--maskImage',required=True,help='path to the MS patient brain mask image')
parser.add_argument('-o', '--outputFolder',required=True,help='path to output folder')
parser.add_argument('-n', '--nbThreads',required=False,type=int,help='Number of execution threads (default: 0 = all cores)',default=0)
parser.add_argument('-c','--consensus',required=False,help='path to consensus image')
parser.add_argument('-p', '--training',action='store_true',help='For training: just execute the preprocessing, ignore core processing and post processing (default: False)')

args=parser.parse_args()

t1Image=args.t1
t2Image=args.t2
flairImage=args.flair
maskImage=args.maskImage
outputFolder=args.outputFolder
cImage=args.consensus
training=args.training
nbThreads=str(args.nbThreads)

if cImage is not None and not(os.path.isfile(cImage)):
    print("IO Error: the file "+cImage+" doesn't exist.")
    quit()

if not(os.path.isfile(t1Image)):
    print("IO Error: the file "+t1Image+" doesn't exist.")
    quit()

if t2Image is not None and not(os.path.isfile(t2Image)):
    print("IO Error: the file "+t2Image+" doesn't exist.")
    quit()

if not(os.path.isfile(flairImage)):
    print("IO Error: the file "+flairImage+" doesn't exist.")
    quit()

if not(os.path.isfile(maskImage)):
    print("IO Error: the file "+maskImage+" doesn't exist.")
    quit()

lesionSegmentation.process(flairImage, t1Image, t2Image, maskImage, cImage, outputFolder, nbThreads, training)