import matplotlib.pyplot as plt
from numpy import ndarray, array

def print_samples(image_matrix, line_labels, cmap='gray', save_name = None):

    if not isinstance(image_matrix, ndarray):
        image_matrix = array(image_matrix)

    rows = image_matrix.shape[0]
    cols = image_matrix.shape[1]
    img_y_size = image_matrix.shape[2]
    img_x_size = image_matrix.shape[3]
    
    vmin = 0.
    vmax = 1.
    
    # see if the image is normalized
    if image_matrix.max() > 1.1:
        vmax = 255
    
    # Adapt the aspect ratio
    aspect_ratio  = rows*img_y_size/(cols*img_x_size)
    figsize = (16, 16*aspect_ratio) if aspect_ratio < 1 else (16/aspect_ratio, 16)
    print(figsize)
    fig, grid = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.1)
    
    for row in range(rows):
        for col in range(cols):
            grid[row,col].axis("off")
            grid[row, col].set_title(f"{line_labels[row]} - {col}")
            grid[row, col].imshow(image_matrix[row, col], cmap=cmap, vmin=vmin , vmax=vmax)  # The AxesGrid object work as a list of axes.

    if save_name:
        plt.savefig(save_name)
    plt.show()

