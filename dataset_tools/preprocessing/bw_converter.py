#!/usr/bin/env python3
"""
Convert all jpeg images in a folder to black & white,
adjust contrast and brightness, and optionally preview interactively.
"""

import cv2
import os
import argparse
import numpy as np

def display_images_in_window(original, gray, adjusted, 
                             window_name: str = "Image Comparison", 
                             title_text: str = "") -> None:
    """Display original, grayscale, and adjusted images side by side in a named window."""
    # Ensure all images are 3-channel for stacking
    if len(gray.shape) == 2:
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray_bgr = gray
    
    # Create comparison view
    stacked = np.hstack((original, gray_bgr, adjusted))
    
    # Add labels
    h, w = stacked.shape[:2]
    labeled = stacked.copy()
    
    # Add title bar if provided
    if title_text:
        cv2.putText(labeled, title_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Add section labels
    section_w = w // 3
    cv2.putText(labeled, "Original", (10, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(labeled, "Grayscale", (section_w + 10, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(labeled, "Adjusted", (2 * section_w + 10, h - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Update the window with new image
    cv2.imshow(window_name, labeled)

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    """Automatically adjust brightness and contrast of an image.
    Clips histogram by given percentage to avoid outliers.
     Args:
         image: Input BGR image
         clip_hist_percent: Percentage of histogram to clip (default: 1%)
     Returns:
         Tuple of (adjusted image, alpha, beta)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def main():
    parser = argparse.ArgumentParser(description="Convert images to black & white and adjust contrast/brightness.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input folder containing jpeg images.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save processed images.")
    parser.add_argument("--save", action="store_true",
                        help="Save the processed images (default: just preview).")
    parser.add_argument("--clip-percent", type=float, default=1.0,
                        help="Percentage of histogram to clip for contrast adjustment (default: 1.0).")
    parser.add_argument("--save-gray", action="store_true",
                        help="Save the grayscale version of the images.")
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    if not os.path.exists(input_folder):
        print("The input folder does not exist.")
        return
    
    if args.save:
        os.makedirs(output_folder, exist_ok=True)

    # Collect files
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print("No images found in folder.")
        return

    print(f"Found {len(files)} images to process")
    
    # Create single window if previewing
    window_name = "Image Comparison - Press SPACE/ENTER for next, Q/ESC to quit"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("\nControls:")
    print("  SPACE/ENTER: Next image")
    print("  Q/ESC: Quit")
    print()

    # Apply to all images
    for i, file_name in enumerate(files):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        img = cv2.imread(input_path)
        if img is None:
            print(f"[{i+1}/{len(files)}] ⚠️  Could not load {file_name}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adjusted, alpha, beta = automatic_brightness_and_contrast(img)

        if not args.save:
            # Show preview
            title = f"[{i+1}/{len(files)}] {file_name} (alpha={alpha:.2f}, beta={beta:.2f})"
            display_images_in_window(img, gray, adjusted, window_name, title)
                
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            print(f"[{i+1}/{len(files)}] ✓ Processed {file_name} (not saved)")

            # Check for quit keys
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("\nExiting...")
                break
            elif key == ord(' ') or key == 13:  # SPACE or ENTER
                pass  # Proceed to next image


        # Save if requested
        else:
            if args.save_gray:
                cv2.imwrite(output_path, gray)
            else:
                cv2.imwrite(output_path, adjusted)
            print(f"[{i+1}/{len(files)}] ✓ Saved {file_name}")
            
    
    # Clean up
    if not args.save:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    if args.save:
        print(f"✅ Processed and saved {len(files)} images to {output_folder}")
    else:
        print(f"✅ Previewed {len(files)} images (use --save to save results)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
