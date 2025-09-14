
detector = ContourTableDetector(
    image_with_text=final_image,
    original_image=zoomed_enlarged,
    table_grid_image=repaired_grid
)
detector.crop_each_bounding_box_and_ocr()
detector.generate_csv_file("output_38_text.csv")
detector.generate_confidence_csv_file("output_38_confidence.csv")

debug_image = detector.get_image_with_final_grid()