import cv2
import matplotlib.pyplot as plt
import os

# Load the image
img = cv2.imread(r"C:\Users\z00511dv\Downloads\DLproj\ocr_app\output\original\region_1.png.")
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot the intensity histogram
plt.figure(figsize=(6, 4))
plt.hist(gray.ravel(), 256, [0, 256], color='k')
plt.title("Intensity Histogram (Grayscale)")
plt.xlabel("Intensity value")
plt.ylabel("Pixel count")

# Save the plot
output_path = os.path.join(output_dir, "intensity_histogram.png")
plt.savefig(output_path)
plt.close()
