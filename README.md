# A Versatile Image Editor Based on Python
**UOIT CSCI 3240U Final Project**  
**By Xuan Zheng and Justin Marsh**  

---

### Introduction  
This project is a comprehensive image editor developed in Python. It offers various functionalities, including brightness and contrast adjustment, color temperature control, object removal, and panorama creation. It uses a user-friendly GUI built with PySimpleGUI.

---

### Required Libraries and Running Conditions  

**Tested Operating System:** Ubuntu 20.04  

**Tested Python Version:** Python 3.10.12  

**Required Libraries:**  
1. PySimpleGUI  
2. NumPy  
3. OpenCV (Tested on Version 4.10)  
4. PIL (Pillow)  
5. `python3-tk` (for GUI support)  
6. `libgl1` (for OpenCV graphical functionality)

To install the required libraries, run the following:  
```bash
sudo apt-get install python3-tk libgl1 -y
pip install PySimpleGUI numpy opencv-python opencv-contrib-python pillow
```  

---

### How to Run the File  

Before proceeding to this step, ensure you have cloned all the files from the repository to your computer, installed all the required libraries using Ubnutu, and opened the correct folder paths.
```bash
cd (path to A-versatile-image-editor-based-on-python-UOIT-CSCI-3240U-FinalProject)
```  
To start the application, use the following command:  
```bash
python3 image_viewer.py IMG_TEST.jpeg
```  

If you want to use a different image for testing, replace `IMG_TEST.jpeg` in the command line with your file (you may need to put the image into that folder first). Or use the app's Load Image function to load the image you want (if you planning to use this please check the Note).

Note: If you want to use the Load Image function to load the image you want. Please ensure the image size of the image you use is larger than or equal to 800*600 pixels. Or the "Selection Tools" might not work as well as expected.

---

### How to Test Each Function  

1. **Load Image:**  
   - Click "Browse" to find the image you want to load.
   - Click "OK" to load the selected image.

2. **Selection Tools:**  
   - Select a region of the image using Rectangle, Circle, or Freehand.
   - For Freehand, you can change the brush size by moving the slider.  
   - You can apply adjustment or object removal for the region you selected.  
   - Note: The selection tool seems to have some issues with resized images (it still works but is offset). If you still see this note, we haven't found a fix.

3. **Adjustment:**  
   - If you have not selected a region, it will be applied to the whole image.   
   - Move those sliders to adjust these properties under the "Adjustments" section.   
   - Click "Apply Adjustments" to finalize changes.

4. **Object Removal:**  
   - Note: This feature is based on basic inpainting and can only achieve nearly seamless results in small areas. It is not so good for large areas.
   - Select a region of the image using the "Selection Tools" section.  
   - Click "Object Removal" to open the inpainting dialog.  
   - Choose the inpainting method (Telea or Navier-Stokes) and adjust the inpainting radius.  
   - Click "Inpaint" to remove the selected object seamlessly.

5. **Filters:**  
   - Select an effect from the "Filters" dropdown (e.g., Sepia, Black and White, Oil Painting).  
   - Click "Apply Filter" to apply the chosen effect to the image.  
   - The filter transforms the image using various artistic or enhancement techniques.  
   - To test multiple filters, reset the image using "Reset Image" and repeat the process.

6. **Panorama Creation:**  
   - Click "Panorama" and use the file dialogue to select (hold ctrl to select multiple files) at least two images (e.g., `IMG_LEFT_BOX.jpeg` and `IMG_RIGHT_BOX.jpeg`).  
   - The system will attempt to stitch the images into a single panorama.  
   - If successful, the panorama will display, and you can save it by clicking "Save Panorama."

7. **Resize Image:**  
   - Click "Resize Image" to open the resizing dialog.  
   - Set the desired width and height or select to maintain the aspect ratio.  
   Note: The maintain aspect ratio function automatically detects the larger value you enter in the input box and scales it to keep the aspect ratio. For example, if the original image size is 800* 600 pixels, when you only change the width to 1000 while choosing Keep Aspect Ratio, the generated image will be 1000* 750 pixels.
   - Select the interpolation method (Nearest Neighbor, Bilinear, or Bicubic).  
   - Click "Resize" to apply the changes. The image will be updated in the editor.

8. **Undo Changes:**  
   - Click "Undo" to revert to the previous state of the image.  
   - The system stores a history of changes, so you can undo multiple steps.

9. **Reset Image:**  
   - Click "Reset Image" to revert the image to its original, unmodified state.  
   - This clears all applied adjustments, filters, or other modifications.

10. **Saving the Image:**  
    - After making desired changes, click "Save Image" and specify the output filename and format.  
    - The processed image will be saved to the chosen location.

---
