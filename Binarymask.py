import cv2
import multiprocessing as mp
import argparse
from imutils import paths
import numpy as np
import os

def chunk(l, n):
	# loop over the list in n-sized chunks
	for i in range(0, len(l), n):
		# yield the current n-sized chunk to the calling function
		yield l[i: i + n]

def process_image(payload):    
    Nonzero_pixels = 0
    for imagePath in payload["input_paths"]:
        print("[INFO] starting process {}".format(payload["id"]))
        img = cv2.imread(imagePath)
        mask = cv2.inRange(img, (200, 200, 200), (255, 255, 255))
        mask_filename = payload["output_path"] + os.path.splitext(os.path.basename(imagePath))[0] + '_mask.png' # Lossless PNG format
        cv2.imwrite(mask_filename, mask)
        print("[INFO] pixels with mask value 255: {} in image".format(cv2.countNonZero(mask)))
        Nonzero_pixels += cv2.countNonZero(mask)
    return Nonzero_pixels

if __name__ == '__main__':
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, type=str, help="path to input directory of images")
    ap.add_argument("-o", "--output", required=True, type=str, help="path to output directory to store output files")
    ap.add_argument("-p", "--procs", type=int, default=-1, help="# of processes to spin up")
    args = vars(ap.parse_args())
    
    #check if the input directory exists
    if not os.path.exists(args["images"]):
        print("[ERROR] path to input directory of images is invalid...")
    
    #if output folder doesn’t exist we create one
    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])
 
    procs = args["procs"] if args["procs"] > 0 else mp.cpu_count()
    proc_IDs = list(range(0, procs))
 
    print("[INFO] Reading image paths...")
    allImages = sorted(list(paths.list_images(args["images"])))
    numImagesPerProc = len(allImages) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))
    
    chunkedPaths = list(chunk(allImages, numImagesPerProc))
    
    payloads = []
    for (i, imagePaths) in enumerate(chunkedPaths):
        outputPath = os.path.sep.join([args["output"], "proc_{}_".format(i)]) # Lossless PNG format
        data = {
			"id": i,
			"input_paths": imagePaths,
			"output_path": outputPath
		}
        payloads.append(data)
        
    print("[INFO] launching pool using {} processes...".format(procs))
    pool = mp.Pool(processes=procs)
    Nonzero_pixels = pool.map(process_image, payloads)
    
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")
    print("[INFO] Total pixels with mask value 255: {}".format(sum(Nonzero_pixels)))
