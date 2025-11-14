import cv2
import numpy as np

def overlay_heatmap(img, heatmap):
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_color = cv2.cvtColor((img*255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    output = cv2.addWeighted(img_color, 0.5, heatmap, 0.5, 0)
    return output
