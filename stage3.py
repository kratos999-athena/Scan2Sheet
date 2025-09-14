import cv2
import numpy as np
import csv
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class ContourTableDetector:

    def __init__(self, image_with_text, original_image, table_grid_image=None):

        self.thresholded_image = image_with_text
        self.original_image = original_image
        self.table_grid_image = table_grid_image
        self.dilated_image = None
        self.contours = None
        self.bounding_boxes = []
        self.rows = []
        self.table_grid = []
        self.mean_height = 0
        self.table = []
        self.confidence_table = []

        self.ocr_engine = None
        if PADDLE_AVAILABLE:
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print(" PaddleOCR engine loaded.")
        else:
            print(" Warning: PaddleOCR is not installed. Tiers 1 & 2 OCR will be disabled.")


        if not TESSERACT_AVAILABLE:
            print(" Warning: Tesseract is not installed. Tier 3 OCR will be disabled.")

        self.super_res_model = None
        self.SUPER_RES_AVAILABLE = False
        try:

            model_path = "EDSR_x4.pb"
            self.super_res_model = cv2.dnn_superres.DnnSuperResImpl_create()
            self.super_res_model.readModel(model_path)
            self.super_res_model.setModel("edsr", 4)
            self.SUPER_RES_AVAILABLE = True
            print(" AI Super-Resolution model loaded successfully.")
        except cv2.error:
            print(f" Warning: Could not load Super-Resolution model. Tier 4 OCR will be disabled.")
            print("   Ensure 'EDSR_x4.pb' is in the correct directory.")

    def _build_grid_from_lines(self):

        if self.table_grid_image is None:
            print("INFO: No table grid image provided. Skipping line-based detection.")
            return

        print("INFO: Attempting Primary Strategy: Detecting cells from grid lines...")

        contours, hierarchy = cv2.findContours(self.table_grid_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cell_boxes = []
        if hierarchy is None:
            print("WARNING: No contour hierarchy found in grid image.")
            return

        for i, cnt in enumerate(contours):

            child_index = hierarchy[0][i][2]

            if child_index != -1:
                current_child_index = child_index

                while current_child_index != -1:

                    x, y, w, h = cv2.boundingRect(contours[current_child_index])

                    if w > 10 and h > 10:
                        cell_boxes.append((x, y, w, h))

                    current_child_index = hierarchy[0][current_child_index][0]

        if not cell_boxes:
            print("WARNING: Primary strategy could not detect any cells from grid lines.")
            return

        sorted_boxes = sorted(cell_boxes, key=lambda box: box[1])

        rows = []
        if not sorted_boxes: return
        current_row = [sorted_boxes[0]]
        for box in sorted_boxes[1:]:

            if abs(box[1] - current_row[-1][1]) < 20:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda box: box[0])

        self.table_grid = rows
        print(f"Primary Strategy Successful: Found a grid with {len(self.table_grid)} rows.")

    def _build_grid_from_text_blobs(self):
        print("INFO: Attempting Fallback Strategy: Detecting cells from text blobs...")

        kernel = np.ones((1, 20), np.uint8)
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel, iterations=3)

        self.contours, _ = cv2.findContours(self.dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.bounding_boxes = []
        for contour in self.contours:
            if cv2.contourArea(contour) > 50:
                self.bounding_boxes.append(cv2.boundingRect(contour))

        heights = [h for (_, _, _, h) in self.bounding_boxes]
        self.mean_height = np.mean(heights) if heights else 0

        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda box: box[1])

        if not self.bounding_boxes: return
        row_tolerance = self.mean_height / 1.5
        self.rows = []
        current_row = [self.bounding_boxes[0]]
        for box in self.bounding_boxes[1:]:
            previous_box_y_center = current_row[-1][1] + current_row[-1][3] / 2
            current_box_y_center = box[1] + box[3] / 2
            if abs(current_box_y_center - previous_box_y_center) < row_tolerance:
                current_row.append(box)
            else:
                self.rows.append(current_row)
                current_row = [box]
        self.rows.append(current_row)

        for i in range(len(self.rows)):
            self.rows[i] = sorted(self.rows[i], key=lambda box: box[0])

        self._split_merged_cells()
        print(f" Fallback Strategy Successful: Found a grid with {len(self.table_grid)} rows.")


    def _split_merged_cells(self):

        if not self.rows: return

        reference_row_index = max(range(len(self.rows)), key=lambda i: len(self.rows[i]))
        reference_row = self.rows[reference_row_index]
        num_columns = len(reference_row)
        column_starts = [box[0] for box in reference_row]

        self.table_grid = []
        for row in self.rows:
            grid_row = [None] * num_columns
            for box in row:
                x, y, w, h = box

                start_col = -1
                for i in range(num_columns - 1, -1, -1):
                    if x >= column_starts[i] - (w * 0.2):
                        start_col = i
                        break
                if start_col == -1: continue
                grid_row[start_col] = box
            self.table_grid.append(grid_row)

    def crop_each_bounding_box_and_ocr(self):
        self._build_grid_from_lines()

        if not self.table_grid:
            print("WARNING: Primary strategy failed. Executing fallback strategy.")
            self._build_grid_from_text_blobs()

        if not self.table_grid:
            print("ERROR: Both detection strategies failed. Could not form a table.")
            return
        print("\nINFO: Starting OCR process on the final grid...")
        for row_idx, grid_row in enumerate(self.table_grid):
            current_text_row = []
            current_confidence_row = []

            for cell_content in grid_row:
                if cell_content is None:
                    current_text_row.append("")
                    current_confidence_row.append("")
                    continue

                x, y, w, h = cell_content
                padding = 5
                cropped_image = self.original_image[max(0, y - padding):y + h + padding, max(0, x - padding):x + w + padding]

                if cropped_image.size > 0:
                    text_result, conf_result = self._get_result_from_paddleocr(cropped_image)
                    if not text_result.strip():
                        text_result, conf_result = self._strong_ocr_pass(cropped_image)
                    if not text_result.strip() and self.SUPER_RES_AVAILABLE:
                        text_result, conf_result = self._super_resolution_ocr_pass(cropped_image)

                    current_text_row.append(text_result.strip())

                    current_confidence_row.append(conf_result if text_result.strip() else "")
                else:
                    current_text_row.append("")
                    current_confidence_row.append("")

            self.table.append(current_text_row)
            self.confidence_table.append(current_confidence_row)
            print(f"   > Processed row {row_idx + 1}/{len(self.table_grid)}")
        print(" OCR process complete.")

    def _get_result_from_paddleocr(self, image):
        if not self.ocr_engine: return "", 0.0
        res = self.ocr_engine.ocr(image, cls=True)
        if not res or not res[0]:
            return "", 0.0
        texts = [line[1][0] for line in res[0]]
        scores = [line[1][1] for line in res[0]]
        full_text = " ".join(texts)
        avg_score = np.mean(scores) if scores else 0.0

        return full_text, avg_score

    def _strong_ocr_pass(self, image):
        print("   INFO: Tier 1 failed, attempting Tier 2 (Stronger Preprocessing)...")
        if image.size == 0 or not self.ocr_engine: return "", 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = max(1.0, 150 / h)
        upscaled = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        denoised = cv2.fastNlMeansDenoising(upscaled, h=30)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrasted = clahe.apply(denoised)
        _, binarized = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._get_result_from_paddleocr(binarized)

    def _get_result_from_tesseract(self, image):
        print("   INFO: Tier 2 failed, attempting Tier 3 (Tesseract)...")
        try:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_img, config=r'--oem 3 --psm 6')
            return text, 0.0
        except Exception: return "", 0.0

    def _super_resolution_ocr_pass(self, image):
        print("   INFO: Tier 3 failed, attempting Tier 4 (AI Super-Resolution)...")
        try:
            sr_image = self.super_res_model.upsample(image)
            return self._strong_ocr_pass(sr_image)
        except Exception as e:
            print(f"   ERROR: Super-resolution pass failed: {e}")
            return "", 0.0

    def generate_csv_file(self, filename="output.csv"):
        """Saves the final extracted table data into a CSV file."""
        with open(filename, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.table)
        print(f"\n Successfully generated text file: {filename}")
    def generate_confidence_csv_file(self, filename="confidence_output.csv"):
        with open(filename, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.confidence_table)
        print(f" Successfully generated confidence score file: {filename}")


    def get_image_with_final_grid(self):
        image_with_boxes = self.original_image.copy()
        for row_idx, grid_row in enumerate(self.table_grid):
            for col_idx, cell_content in enumerate(grid_row):
                if isinstance(cell_content, tuple) and len(cell_content) == 4:
                    x, y, w, h = cell_content

                    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    cv2.putText(image_with_boxes, f"{row_idx},{col_idx}", (x + 5, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return image_with_boxes