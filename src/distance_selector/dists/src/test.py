import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image
image_path = r"output\Figure_2.png"
img = mpimg.imread(image_path)

# Coordinates to mark (x = frame numbers, y will be estimated visually since we only have x)
frame_points = [86, 128, 167, 202, 223, 307, 337, 376, 421, 440, 510, 701, 775, 806, 842]
# [169, 226, 234, 239, 244, 379, 421, 482, 571, 584, 618, 654, 691, 704, 717, 818]
# Display the image
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(img)

# Estimate y-coordinates (relative to the graph area in the image)
# Since we only have x values, we draw vertical lines at these frame positions (approximate)
for x in frame_points:
    # Map frame number to image x-coordinate (image width ~1060, frame range ~0-900)
    x_img = int((x / 900) * img.shape[1])
    ax.axvline(x=x_img, color='red', linestyle='--', linewidth=1)

plt.axis("off")
plt.show()
