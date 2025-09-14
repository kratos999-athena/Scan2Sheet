import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def repair_table_with_projection(image_with_white_lines):
    _, binary_image = cv2.threshold(image_with_white_lines, 127, 255, cv2.THRESH_BINARY)
    h, w = binary_image.shape
    horizontal_projection = np.sum(binary_image, axis=0) // 255
    peaks_x, _ = find_peaks(horizontal_projection, prominence=h*0.2)


    vertical_projection = np.sum(binary_image, axis=1) // 255


    peaks_y, _ = find_peaks(vertical_projection, prominence=w*0.2)


    reconstructed_grid = np.zeros_like(binary_image)
    line_color = (255, 255, 255)
    line_thickness = 2

    for x in peaks_x:
        cv2.line(reconstructed_grid, (x, 0), (x, h), line_color, line_thickness)

    for y in peaks_y:
        cv2.line(reconstructed_grid, (0, y), (w, y), line_color, line_thickness)

    return reconstructed_grid, horizontal_projection, vertical_projection, peaks_x, peaks_y


input_image = thickened_table
repaired_grid, hp, vp, px, py = repair_table_with_projection(input_image)

plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input: Broken Structure')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(repaired_grid, cmap='gray')
plt.title('Output: Repaired with Projections')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(hp)
plt.plot(px, hp[px], "x", color='red')
plt.title('Horizontal Projection (finding Vertical Lines)')
plt.xlabel('Column (X-coordinate)')
plt.ylabel('Pixel Sum')

plt.subplot(2, 2, 4)
plt.plot(vp)
plt.plot(py, vp[py], "x", color='red')
plt.title('Vertical Projection (finding Horizontal Lines)')
plt.xlabel('Row (Y-coordinate)')
plt.ylabel('Pixel Sum')

plt.tight_layout()
plt.show()