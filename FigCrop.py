# import the necessary packages
import argparse
from pathlib import Path
import numpy as np
import cv2
import sys
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

orange = (20, 150, 240,255)
light_blue = (242, 135, 63, 255)

line_color = light_blue
default_roi_scale = 3.0
default_image_scale = 1.0
default_line_thickness = 5 # in px
default_offset = 0 # in px towards the orig image

def getColor(index):
    orange = np.uint8([[[20, 150, 240]]])
    hsv_orange = cv2.cvtColor(orange,cv2.COLOR_BGR2HSV)
    next_hue = hsv_orange
    next_hue[0][0][0] = (next_hue[0][0][0] + round(index*2.23606*373))%255 #some magic number hihi
    next_rgb = cv2.cvtColor(next_hue,cv2.COLOR_HSV2BGR)
    return (int(next_rgb[0][0][0]),int(next_rgb[0][0][1]),int(next_rgb[0][0][2]),255)


def write_image(fig, image_name, args):
        # get putput path
    img_path = Path(image_name)
    img_dir = img_path.parent
    img_filename = img_path.name
    
    out = str("./fig_"+str(img_filename))
    if args["output"] is not None:
        out_path = Path(args["output"])
        if out_path.is_dir():
            out = str(out_path/Path("fig_"+str(img_filename)))
        else:
            if len(args["input"]) > 1:
                print("WARNING: multiple files will be written to same file location")
            out = str(out_path)
            if not out_path.suffix:
                out += '.png'

    # write out the figure
    print("write out "+out)
    cv2.imwrite(out, fig) 

def rescale_images(image, roi, bbox, roi_scale=None, roi_width=None, image_scale=default_image_scale):
     # resize image
    width = int(image.shape[1] * image_scale)
    height = int(image.shape[0] * image_scale)
    dim = (width, height)
    scaled_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 

    # resize roi
    if roi_scale is not None and roi_width is not None:
        print('dont use scale and width simultaneously DUMBASS')
        exit()
    elif roi_scale is not None:
        width = int(roi.shape[1] * roi_scale)
        height = int(roi.shape[0] * roi_scale)
    elif roi_width is not None:
        width = roi_width
        height = int(roi.shape[0] * float(roi_width)/float(roi.shape[1]))
    else:
        print('set scale OR width DUMBASS')
        exit()

    dim = (width, height)
    scaled_roi= cv2.resize(roi, dim, interpolation = cv2.INTER_NEAREST) 

    # resize bbox region
    scaled_bbox = [(int(image_scale*float(bbox_elem0)),int(image_scale*float(bbox_elem1))) for (bbox_elem0, bbox_elem1) in bbox]

    return scaled_image, scaled_roi, scaled_bbox

def layout_right(scaled_image, scaled_roi, scaled_bbox, line_thickness=default_line_thickness, offset=default_offset):
    # init fig canvas
    fig_height = max(scaled_image.shape[0], scaled_roi.shape[0])
    fig_width = max(scaled_image.shape[1], scaled_image.shape[1]+scaled_roi.shape[1]-offset)
    if scaled_image.shape[2] == 4: # #RGBA, BGRA whatever
        # init transparent
        fig = np.full((fig_height, fig_width, scaled_image.shape[2]), fill_value=0, dtype=np.uint8)
    else: #RGB, BGR, grayscale whatever 
        # init white
        fig = np.full((fig_height, fig_width, scaled_image.shape[2]), fill_value=255, dtype=np.uint8)

    #paste image
    fig[0:scaled_image.shape[0],0:scaled_image.shape[1],:] = scaled_image
    # draw a rectangle around the roi
    cv2.rectangle(fig, scaled_bbox[0], scaled_bbox[1], line_color, line_thickness)

    #paste roi
    roi_paste_start = [int(fig_height/2 - scaled_roi.shape[0]/2), scaled_image.shape[1]-offset]
    roi_paste_end = [roi_paste_start[0]+scaled_roi.shape[0], roi_paste_start[1]+scaled_roi.shape[1]]
    fig[roi_paste_start[0]:roi_paste_end[0], roi_paste_start[1]:roi_paste_end[1],:] = scaled_roi
    # draw a rectangle around the the scaled, pasted region of interest
    cv2.rectangle(fig, (roi_paste_start[1]+line_thickness, roi_paste_start[0]+line_thickness), 
                        (roi_paste_end[1]-line_thickness, roi_paste_end[0]-line_thickness), 
                        line_color, line_thickness) 

    # draw lines for "zoom" effect
    x0, y0 = scaled_bbox[0]
    x1, y1 = scaled_bbox[1]
    line_start = (min(x0,x1), min(y0,y1))
    line_end = (roi_paste_start[1]+line_thickness, roi_paste_start[0]+line_thickness)
    cv2.line(fig, line_start, line_end, line_color, line_thickness//2,cv2.LINE_AA)
    
    line_start = (min(x0,x1), max(y0,y1))
    line_end = (roi_paste_start[1]+line_thickness, roi_paste_end[0]-line_thickness)
    cv2.line(fig, line_start, line_end, line_color, line_thickness//2,cv2.LINE_AA)

    return fig

def layout_sqr_bottom(tile, scaled_image, scaled_roi, scaled_bbox, line_thickness=default_line_thickness):
    if( tile == 0):
        # init fig canvas
        img_height = scaled_image.shape[0]
        fig_height = img_height+ scaled_roi.shape[0]+line_thickness
        fig_width = scaled_image.shape[1]
        if scaled_image.shape[2] == 4: # #RGBA, BGRA whatever
            # init transparent
            fig = np.full((fig_height, fig_width, scaled_image.shape[2]), fill_value=0, dtype=np.uint8)
        else: #RGB, BGR, grayscale whatever 
            # init white
            fig = np.full((fig_height, fig_width, scaled_image.shape[2]), fill_value=255, dtype=np.uint8)

        #paste image
        fig[0:img_height,0:scaled_image.shape[1],:] = scaled_image
    else:
        fig = scaled_image
        img_height = fig.shape[0]- scaled_roi.shape[0]-line_thickness
    # draw a rectangle around the roi
    color = getColor(tile)
    cv2.rectangle(fig, scaled_bbox[0], scaled_bbox[1], color, line_thickness)


    #paste roi
    roi_paste_start = [img_height, (tile*scaled_roi.shape[1])]
    roi_paste_end = [roi_paste_start[0]+scaled_roi.shape[0], roi_paste_start[1]+scaled_roi.shape[1]]
    fig[roi_paste_start[0]:roi_paste_end[0], roi_paste_start[1]:roi_paste_end[1],:] = scaled_roi
    # draw a rectangle around the the scaled, pasted region of interest
    cv2.rectangle(fig, (roi_paste_start[1]+line_thickness, roi_paste_start[0]+line_thickness), 
                        (roi_paste_end[1]-line_thickness, roi_paste_end[0]-line_thickness), 
                        color, line_thickness) 
    return fig

def layout_sqr_right(tile, scaled_image, scaled_roi, scaled_bbox, line_thickness=default_line_thickness):
    if( tile == 0):
        # init fig canvas
        img_width = scaled_image.shape[1]
        fig_width = img_width+ scaled_roi.shape[1]+line_thickness
        fig_height = scaled_image.shape[0]
        if scaled_image.shape[2] == 4: # #RGBA, BGRA whatever
            # init transparent
            fig = np.full((fig_height, fig_width, scaled_image.shape[2]), fill_value=0, dtype=np.uint8)
        else: #RGB, BGR, grayscale whatever 
            # init white
            fig = np.full((fig_height, fig_width, scaled_image.shape[2]), fill_value=255, dtype=np.uint8)

        #paste image
        fig[0:scaled_image.shape[0],0:img_width,:] = scaled_image
    else:
        fig = scaled_image
        img_width = fig.shape[1]- scaled_roi.shape[1]-line_thickness
    # draw a rectangle around the roi
    color = getColor(tile)
    cv2.rectangle(fig, scaled_bbox[0], scaled_bbox[1], color, line_thickness)


    #paste roi
    roi_paste_start = [(tile*scaled_roi.shape[0]), img_width]
    roi_paste_end = [roi_paste_start[0]+scaled_roi.shape[0], roi_paste_start[1]+scaled_roi.shape[1]]
    fig[roi_paste_start[0]:roi_paste_end[0], roi_paste_start[1]:roi_paste_end[1],:] = scaled_roi
    # draw a rectangle around the the scaled, pasted region of interest
    cv2.rectangle(fig, (roi_paste_start[1]+line_thickness, roi_paste_start[0]+line_thickness), 
                        (roi_paste_end[1]-line_thickness, roi_paste_end[0]-line_thickness), 
                        color, line_thickness) 
    return fig

def click_and_crop(event, x, y, flags, image):
	# grab references to the global variables
	global refPt, cropping, line_color
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append((x, y))
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[-2], refPt[-1], line_color, 2)
		cv2.imshow("image", image)
def FigCrop(rawArgs = None):
    global refPt
    try:
        print('hi')
        args = []
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        # "Example call: python .\FigCrop.py -i .\FigCropInput\test_input.png .\FigCropInput\test_input2.png -o .\FigCropOutput.png -l sqr_bottom"
        ap.add_argument("-i", "--input", required=True, nargs='+', help="Path to input image(s)")
        ap.add_argument("-o", "--output", required=False, help="output file(s) or directory")
        ap.add_argument("-is", "--image_scale", required=False, help="image scale factor")
        ap.add_argument("-l", "--layout", required=False, help="""Choose layout from (default) 'right' or 
                                                                'sqr_bottom' (NOTE: when using squares bottom, 
                                                                all roi scaling and offset params (rs, rw, off) are ignored)""")
        ap.add_argument("-rs", "--roi_scale", required=False, help="roi factor scale factor")
        ap.add_argument("-rw", "--roi_width", required=False, help="pasted roi width, will be ignored if scale is set too")
        ap.add_argument("-off", "--offset", required=False, help="offset of pasted, zoomed roi towards the image center")

        if rawArgs is None:
            # when called from command line
            args = vars(ap.parse_args(rawArgs))
        else:
            args = vars(ap.parse_args())
        

        # check if input is dir, 
        # if so replace args[input] by a list
        # containing all the images in the dir
        
        if len(args["input"]) == 1:
            dir = Path(args["input"][0])
            if dir.is_dir():
                image_list = [str(path) for path in dir.glob("*.png")]
                image_list.extend([str(path) for path in dir.glob("*.jpg")])
                args["input"] = image_list

        
        # load the image, clone it, and setup the mouse callback function
        image = cv2.imread(args["input"][0])
        clone = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop, image)
        
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF
        
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()
                refPt = []
        
            # if the 'd' key is pressed, break from the loop
            elif key == ord("d"):
                break

        # set image scale
        image_scale = default_image_scale
        if args["image_scale"] is not None:
            image_scale = float(args["image_scale"])

        #save image size of first one
        img_dims = image.shape
        
         # choose layout
        if args["layout"] == None or args["layout"] == "right":
            # if there != two reference points, then exit
            if len(refPt) != 2:
                print("no roi or too many selected")
                exit()

            # set scale params and offset
            roi_scale = default_roi_scale
            roi_width = None
            if args["roi_scale"] is not None and args["roi_width"] is not None:
                print('set scale OR width DUMBASS')
                exit()
            if args["roi_scale"] is not None:
                roi_scale = float(args["roi_scale"])
            elif args["roi_width"] is not None:
                roi_scale = None
                roi_width = int(args["roi_width"])
            offset = default_offset
            if args["offset"] is not None:
                offset = int(args["offset"])

            for image_name in args["input"]: 
                # load the image, clone it, and setup the mouse callback function
                clone = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                if clone.shape[0] != img_dims[0] or clone.shape[1] != img_dims[1]:
                    img_dims
                    print("mismatching resolution, skip "+image_name)
                    print(str(clone.shape) +"!="+str(img_dims))
                    continue
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                # cv2.imshow("ROI", roi)

                # scale image accordingly
                scaled_thingies = rescale_images(clone, roi, refPt, 
                                    image_scale=image_scale, roi_scale=roi_scale, roi_width=roi_width)
                scaled_image, scaled_roi, scaled_bbox = scaled_thingies
                fig = layout_right(scaled_image, scaled_roi,scaled_bbox,
                                    offset=offset)
                
                # show and write
                cv2.imshow("Figure", fig)
                write_image(fig, image_name, args)

                cv2.waitKey(30)

        elif args["layout"] == "sqr_one_right":
            if len(refPt) != 2:
                print("no roi or too many selected")
                exit()
               
            for image_name in args["input"]: 
                # load the image, clone it, and setup the mouse callback function
                clone = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                # skip images with mismathin resolution
                if clone.shape[0] != img_dims[0] or clone.shape[1] != img_dims[1]:
                    img_dims
                    print("mismatching resolution, skip "+image_name)
                    continue
                for ri in range(0,len(refPt), 2):
                    roi_in_w = refPt[1][0]- refPt[0][0]
                    roi_in_h = refPt[1][1]- refPt[0][1]
                    in_dim = max(roi_in_w, roi_in_h)
                    
                    refPt[1] = (refPt[0][0]+in_dim, refPt[0][1]+in_dim)
                    
                    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                    print(in_dim)
                    roi_width = round(clone.shape[1]*image_scale)
                    offset = 0
                    # cv2.imshow("ROI", roi)

                    # scale image accordingly
                    scaled_thingies = rescale_images(clone, roi, refPt, 
                                        image_scale=image_scale, roi_width=roi_width)
                    scaled_image, scaled_roi, scaled_bbox = scaled_thingies
                    fig = layout_right(scaled_image, scaled_roi,scaled_bbox,
                                        offset=offset)
                
                # show and write
                cv2.imshow("Figure", fig)
                write_image(fig, image_name, args)

                cv2.waitKey(30)
        elif args["layout"] == "sqr_bottom":
            num_rois = len(refPt)//2
            if num_rois < 1:
                print("no enugh roi selected")
                exit()
                
            for image_name in args["input"]: 
                # load the image, clone it, and setup the mouse callback function
                clone = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                if clone.shape[0] != img_dims[0] or clone.shape[1] != img_dims[1]:
                    img_dims
                    print("mismatching resolution, skip "+image_name)
                    continue
                fig = None
                for ri in range(0,len(refPt), 2):
                    # make roi squared with max dim
                    roi_in_w = refPt[ri+1][0]- refPt[ri+0][0]
                    roi_in_h = refPt[ri+1][1]- refPt[ri+0][1]
                    in_dim = max(roi_in_w, roi_in_h)
                    refPt[ri+1] = (refPt[ri+0][0]+in_dim, refPt[ri+0][1]+in_dim)
                    
                    roi = clone[refPt[ri+0][1]:refPt[ri+1][1], refPt[ri+0][0]:refPt[ri+1][0]]
                    roi_width = int((clone.shape[1]*image_scale)//num_rois)
                    
                    # scale image accordingly
                    scaled_thingies = rescale_images(clone, roi, [refPt[ri+0],refPt[ri+1]], 
                                        image_scale=image_scale, roi_width=roi_width)
                    scaled_image, scaled_roi, scaled_bbox = scaled_thingies
                    # first call inits canvas n stuff, after wards only new boxes are added
                    if fig is None:
                        fig = layout_sqr_right(ri//2, scaled_image, scaled_roi,scaled_bbox)
                    else:
                        fig = layout_sqr_right(ri//2, fig, scaled_roi,scaled_bbox)
                
                # show and write
                cv2.imshow("Figure", fig)
                write_image(fig, image_name, args)

                cv2.waitKey(3000)
        elif args["layout"] == "sqr_right":
            num_rois = len(refPt)//2
            if num_rois < 1:
                print("no enugh roi selected")
                exit()
                
            for image_name in args["input"]: 
                # load the image, clone it, and setup the mouse callback function
                clone = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                if clone.shape[0] != img_dims[0] or clone.shape[1] != img_dims[1]:
                    img_dims
                    print("mismatching resolution, skip "+image_name)
                    continue
                fig = None
                for ri in range(0,len(refPt), 2):
                    # make roi squared with max dim
                    roi_in_w = refPt[ri+1][0]- refPt[ri+0][0]
                    roi_in_h = refPt[ri+1][1]- refPt[ri+0][1]
                    in_dim = max(roi_in_w, roi_in_h)
                    refPt[ri+1] = (refPt[ri+0][0]+in_dim, refPt[ri+0][1]+in_dim)
                    
                    roi = clone[refPt[ri+0][1]:refPt[ri+1][1], refPt[ri+0][0]:refPt[ri+1][0]]
                    roi_width = int((clone.shape[0]*image_scale)//num_rois)
                    
                    # scale image accordingly
                    scaled_thingies = rescale_images(clone, roi, [refPt[ri+0],refPt[ri+1]], 
                                        image_scale=image_scale, roi_width=roi_width)
                    scaled_image, scaled_roi, scaled_bbox = scaled_thingies
                    # first call inits canvas n stuff, after wards only new boxes are added
                    if fig is None:
                        fig = layout_sqr_right(ri//2, scaled_image, scaled_roi,scaled_bbox)
                    else:
                        fig = layout_sqr_right(ri//2, fig, scaled_roi,scaled_bbox)
                
                # show and write
                cv2.imshow("Figure", fig)
                write_image(fig, image_name, args)

                cv2.waitKey(3000)
        else:
            print("layout not found")

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        exit()

    # close all open windows
    cv2.destroyAllWindows()

# TODO:
# match pasted roi height with fig height?
# use all args

if __name__ == '__main__':
    print('hi')
    FigCrop()