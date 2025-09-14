grayscaleimage=cv2.cvtColor(zoomed_enlarged,cv2.COLOR_RGB2GRAY)## pdf2image always extract the image in RGB form and not in BGR form as used standardly
_, thresh_otsu = cv2.threshold(grayscaleimage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)## otstu method is best for usage when yhe image quality is something we are not sur about it automatically assigns the best possible threshold setting
invert=cv2.bitwise_not(thresh_otsu)
plt.imshow(invert,cmap="gray")
plt.show()
extractor = LineExtractor(invert)
extractor.erode_vertical_lines()
extractor.erode_horizontal_lines()
extractor.combine_eroded_images()

# Show result
plt.imshow(extractor.vertical_lines_eroded_image, cmap='gray')
plt.title("Detected Vertical Lines")
plt.axis("off")
plt.show()
plt.imshow(extractor.horizontal_lines_eroded_image, cmap='gray')
plt.show()
plt.imshow(extractor.combined_image, cmap='gray')
plt.show()

kernel = np.ones((3, 3), np.uint8)
thickened_table = cv2.dilate(extractor.combined_image, kernel, iterations=2)
plt.imshow(thickened_table, cmap='gray')
plt.show()

subtracted_image=cv2.subtract(extractor.inverted_image,thickened_table)
plt.imshow(subtracted_image,cmap="gray")
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
subtracted_imagee=cv2.erode(subtracted_image,kernel,iterations=1)
final_image=cv2.dilate(subtracted_imagee,kernel,iterations=1)

plt.imshow(final_image,cmap="gray")
plt.show()

