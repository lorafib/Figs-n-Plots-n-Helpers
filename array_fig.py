from pathlib import Path
import cv2
import numpy as np

input_path = './input_lightfield_grid/'

image_name = 'shot'
image_width = 120
image_height = 100

file_ending = '.png'
grid_w = 7
grid_h = 7

def main():
	print(input_path)
	c = 0
	lfgrid = np.zeros([image_height*grid_h, image_width*grid_w, 3])
	for grid_y in range(grid_h):
		for grid_x in range(grid_w-1, -1, -1):
			lin_index = grid_y *grid_h+grid_x+1

			# read image
			full_image_path = input_path + image_name + str(lin_index)+ file_ending
			
			# img = np.full([image_height, image_width, 3], c/grid_h*grid_w)
			# cv2.imwrite(full_image_path, img)
			
			img = cv2.imread(full_image_path)

			# paste image into array
			lf_x = grid_x *image_width
			lf_y = grid_y *image_height
			lfgrid[lf_y:lf_y+image_height, lf_x:lf_x+image_width, :] = img

			c+=1
			
	full_lf_path = 'lightfield_array_fig'+ file_ending
	cv2.imwrite(full_lf_path, lfgrid)


if __name__ == '__main__':
	main()