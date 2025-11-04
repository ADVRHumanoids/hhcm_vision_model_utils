# Image Preprocessing Tools

This directory contains tools for preprocessing and enhancing images before annotation or training.

## Tools

### bw_converter.py

Convert images to black & white with automatic histogram-based contrast and brightness adjustment.

**Purpose**: Batch process images for grayscale conversion with intelligent automatic enhancement, or preview adjustments interactively before committing.

**Algorithm**:
The automatic contrast/brightness adjustment uses histogram clipping to avoid outliers:

1. **Histogram Analysis**: Compute cumulative histogram of grayscale values
2. **Outlier Clipping**: Clip specified percentage from both dark and bright ends
3. **Range Stretching**: Stretch remaining range to full 0-255 dynamic range
4. **Linear Transformation**: Apply `output = alpha * input + beta`
   - `alpha`: Contrast factor (controls slope)
   - `beta`: Brightness offset (controls baseline)

This method is more robust than simple min-max stretching as it ignores extreme outliers (very dark or very bright pixels that would skew the adjustment).

**Arguments**:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--input` | str | Yes | - | Path to input folder containing images |
| `--output` | str | Yes | - | Path to save processed images |
| `--save` | flag | No | False | Enable saving mode (default: preview only) |
| `--clip-percent` | float | No | 1.0 | Histogram clip percentage (higher = more aggressive) |
| `--save-gray` | flag | No | False | Save grayscale only without contrast adjustment |

**Preview Mode Controls**:

| Key | Action |
|-----|--------|
| SPACE / ENTER | Next image |
| Q / ESC | Quit viewer |

**Requirements**:
```bash
pip install opencv-python numpy
```

**Usage Examples**:

```bash
# Preview images with automatic adjustment (no saving)
python3 bw_converter.py --input raw_images/ --output processed/

# Batch process and save with default settings
python3 bw_converter.py \
    --input raw_images/ \
    --output processed/ \
    --save

# Save grayscale only without contrast adjustment
python3 bw_converter.py \
    --input raw_images/ \
    --output grayscale_only/ \
    --save \
    --save-gray

# Custom histogram clipping for more aggressive enhancement
python3 bw_converter.py \
    --input low_contrast_images/ \
    --output enhanced/ \
    --save \
    --clip-percent 2.5

# Preview first, then save after confirming quality
# Step 1: Preview (press SPACE to browse)
python3 bw_converter.py --input raw/ --output processed/
# Step 2: Save after confirming
python3 bw_converter.py --input raw/ --output processed/ --save
```


> ### **What is histogram clipping?** 
>
> Histogram clipping removes extreme outliers from the pixel intensity distribution before contrast stretching. This prevents a few very dark or very bright pixels from dominating the adjustment.  
> **Example:**  
> - 98% of pixels in range [30-200]  
> - 1% very dark pixels [0-10]  
> - 1% very bright pixels [245-255]  
>
> **Without clipping** (simple min-max):  
> - Stretch [0-255] → [0-255] (no change, poor contrast)  
>
> **With 1% clipping:**  
> - Clip outliers, stretch [30-200] → [0-255] (significant improvement)  
>
> **Clip Percentage Guidelines:**  
>
> | Clip % | Use Case | Effect |
> |--------|----------|--------|
> | 0.5%   | Subtle enhancement | Minimal change, preserves original look |
> | 1.0%   | Default (recommended) | Balanced enhancement for most images |
> | 2.0%   | Moderate enhancement | Noticeable improvement, good for low contrast |
> | 3.0%+  | Aggressive enhancement | Strong adjustment, may lose detail in shadows/highlights |

---

## Dependencies

```bash
pip install opencv-python>=4.5.0
pip install numpy>=1.19.0
```

---

## Support and Resources

- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)
- **Histogram Equalization**: [https://en.wikipedia.org/wiki/Histogram_equalization](https://en.wikipedia.org/wiki/Histogram_equalization)
- **Contrast Stretching**: [https://en.wikipedia.org/wiki/Normalization_(image_processing)](https://en.wikipedia.org/wiki/Normalization_(image_processing))
