# Handwriting-OCR-for-Graphology

This piece of code works with an image of Hebrew handwritten text. It then uses a process of normalization to output a numerical value between 0 and 1 for specific graphological features. The output is therefore quantitative and objective data that can be used for professional analysis. Though specific individuals contributed to the development of the features, the data normalization aspect was a team effort. 

Deterministic computer vision techniques are used in the system. These include the use of the HoughLines function in the OpenCV library for the detection of lines , the distanceTransform function for the measurement of thickness , and the use of morphological techniques for the determination of baselines .

Currently, the system can perform the following core features:

    Slant (Developed by Shahar Lankri): Computes the angles of the writing using a combination of global shear transformations and local geometric moments.

    Stroke Thickness (Developed by Ran Uram): Measures the "ink radius" of the text using distance transformations .

    Baseline Alignment (Developed by Daniel Giron): Identifies the printed line and the extent to which the text "floats" in relation to the printed lines.

A set of structured data has been created with authentic handwritten data that can be used as the ground truth in the future.
