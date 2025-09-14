import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter
from scipy.ndimage import rotate
import pytesseract

def correct_orientation_advanced_cv(image: np.ndarray) -> np.ndarray:

    print("INFO (Initial Check): Applying advanced text line analysis for orientation...")

    confidences = []
    angles = [0, 90, 180, 270]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for angle in angles:
        if angle == 90: test_img = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180: test_img = cv2.rotate(gray, cv2.ROTATE_180)
        elif angle == 270: test_img = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: test_img = gray

        _, thresh = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        h_proj = np.sum(thresh, axis=1)
        peaks = np.where(h_proj > np.mean(h_proj))[0]

        if len(peaks) < 5:
            confidences.append(0)
            continue

        lines, start = [], peaks[0]
        for i in range(1, len(peaks)):
            if peaks[i] != peaks[i-1] + 1:
                lines.append((start, peaks[i-1]))
                start = peaks[i]
        lines.append((start, peaks[-1]))

        line_asymmetries = []
        for start_y, end_y in lines:
            if end_y - start_y < 5: continue

            line_img = thresh[start_y:end_y, :]
            h, _ = line_img.shape
            midpoint = h // 2
            top_half_sum = np.sum(line_img[0:midpoint, :])
            bottom_half_sum = np.sum(line_img[midpoint:, :])

            if bottom_half_sum > 0 and top_half_sum > 0:
                line_asymmetries.append(bottom_half_sum > top_half_sum)

        if line_asymmetries:
            confidences.append(np.mean(line_asymmetries))
        else:
            confidences.append(0)

    best_angle = angles[np.argmax(confidences)]
    print(f"INFO (Initial Check): Best orientation angle found: {best_angle} degrees with {max(confidences):.2%} confidence.")

    if best_angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif best_angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
    elif best_angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def verify_and_correct_orientation_tesseract(image: np.ndarray) -> np.ndarray:
    """
    The 'Sanity Check': Uses Tesseract OSD on a downscaled image for speed.
    """
    print("INFO (Sanity Check): Using Tesseract OSD to verify and finalize orientation...")
    try:
        h, w = image.shape[:2]
        target_width = 1500
        if w > target_width:
            scale = target_width / w
            small_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            print(f"INFO (Sanity Check): Image resized for OSD from {w}px to {target_width}px width.")
        else:
            small_img = image

        osd = pytesseract.image_to_osd(small_img, output_type=pytesseract.Output.DICT)
        rotation = osd['rotate']

        if rotation != 0:
            print(f"INFO (Sanity Check): Tesseract detected a final {rotation}-degree correction is needed.")
            
            if rotation == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180: return cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        print("INFO (Sanity Check): Tesseract confirms orientation is correct.")
        return image

    except Exception as e:
        print(f"ERROR: Tesseract OSD sanity check failed: {e}")
        return image

def preprocess_image(image: np.ndarray, zoom_factor: float = 2.0):

    initial_orientation = correct_orientation_advanced_cv(image)
    img_oriented = verify_and_correct_orientation_tesseract(initial_orientation)


    print("INFO: Starting Stage 2: Fine-Tuning Skew...")
    gray_oriented = cv2.cvtColor(img_oriented, cv2.COLOR_BGR2GRAY)
    thresh_oriented = cv2.threshold(gray_oriented, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    print("INFO: Skew Correction - Coarse search...")
    coarse_scores = []
    coarse_angles = np.arange(-5, 5.1, 1)
    for angle in coarse_angles:
        rotated = inter.rotate(thresh_oriented, angle, reshape=False, order=0)
        projection = np.sum(rotated, axis=1)
        coarse_scores.append((np.var(projection), angle))

    _, coarse_angle = max(coarse_scores, key=lambda x: x[0])

    fine_search_start = coarse_angle - 1
    fine_search_end = coarse_angle + 1
    print(f"INFO: Skew Correction - Fine search between {fine_search_start}° and {fine_search_end}°...")

    fine_scores = []
    fine_angles = np.arange(fine_search_start, fine_search_end + 0.1, 0.1)
    for angle in fine_angles:
        rotated = inter.rotate(thresh_oriented, angle, reshape=False, order=0)
        projection = np.sum(rotated, axis=1)
        fine_scores.append((np.var(projection), angle))

    _, best_angle = max(fine_scores, key=lambda x: x[0])

    print(f"INFO: Best angle for skew correction: {best_angle:.2f} degrees.")
    (h, w) = img_oriented.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    deskewed = cv2.warpAffine(img_oriented, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

    print("INFO: Starting Stage 3: Cropping to Content...")
    deskewed_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    _, deskewed_thresh = cv2.threshold(deskewed_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(deskewed_thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("WARNING: No contours found for cropping. Returning full deskewed image.")
        return deskewed

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    tight_crop = deskewed[y:y+h, x:x+w]

    final_result = cv2.resize(tight_crop, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    return final_result

image_path = "path/to/your/image.jpg"
image_bgr = cv2.rotate(np.array(images[1]),cv2.ROTATE_90_CLOCKWISE)

if image_bgr is None:
    print(f"ERROR: Could not load image from path: {image_path}")
else:
    zoomed_enlarged = preprocess_image(image_bgr, zoom_factor=2.0)

    if zoomed_enlarged is not None:
        print("\n Preprocessing complete.")
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(zoomed_enlarged, cv2.COLOR_BGR2RGB))
        plt.title("Final Preprocessed Result")
        plt.axis("off")
        plt.show()