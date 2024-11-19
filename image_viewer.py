import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os

def np_im_to_data(im):
    """
    Convert a NumPy image array to a PNG format that can be displayed in PySimpleGUI.
    """
    im = Image.fromarray(im)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust brightness and contrast of the image.
    """
    adjusted = cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)
    return adjusted

def adjust_color_temperature(image, temp_value):
    """
    Adjust the color temperature of the image.
    """
    temp_mapped = temp_value * (40 / 100)
    image = image.astype(np.float32)
    b, g, r = cv2.split(image)
    if temp_mapped >= 0:
        r += (temp_mapped * 2.55)
        b -= (temp_mapped * 2.55)
    else:
        temp_mapped = -temp_mapped
        b += (temp_mapped * 2.55)
        r -= (temp_mapped * 2.55)
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    adjusted = cv2.merge([b, g, r]).astype(np.uint8)
    return adjusted

def sharpen_image(image, amount=1):
    """
    Sharpen the image.
    """
    amount_mapped = (amount - 1) * (5 / 99)
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + amount_mapped, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def blur_image(image, amount=1):
    """
    Blur the image.
    """
    ksize = int(amount) * 2 + 1
    if ksize < 1:
        ksize = 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred

def apply_filter(image, filter_type):
    """
    Apply a selected filter to the image.
    """
    if filter_type == 'Sepia':
        filtered = cv2.transform(image, np.matrix([[0.272, 0.534, 0.131],
                                                   [0.349, 0.686, 0.168],
                                                   [0.393, 0.769, 0.189]]))
    elif filter_type == 'Black and White':
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif filter_type == 'Oil Painting':
        filtered = cv2.xphoto.oilPainting(image, 7, 1)
    elif filter_type == 'Oil Painting 2':
        filtered = oil_painting_2_filter(image)
    elif filter_type == 'Film Effect':
        filtered = film_effect(image)
    else:
        filtered = image
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered

def oil_painting_2_filter(image):
    """
    Apply the Oil Painting 2 filter to the image.
    """
    processed_image = draw_strokes(image)
    edges = edge_detection(processed_image)
    processed_image = clip_strokes(processed_image, edges)
    return processed_image

def film_effect(image):
    """
    Apply a film effect to the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = np.random.normal(0, 10, image.shape[:2])
    noisy_image = blurred + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    filtered = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2RGB)
    return filtered

def draw_strokes(image, stroke_width_range=(1, 3), stroke_length_range=(1, 9)):
    """
    Draw strokes on the image to create an oil painting effect.
    """
    output_image = image.copy()
    h, w, _ = image.shape
    stroke_widths = np.random.randint(stroke_width_range[0], stroke_width_range[1], size=(h, w))
    stroke_lengths = np.random.randint(stroke_length_range[0], stroke_length_range[1], size=(h, w))
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            stroke_length = stroke_lengths[y, x]
            stroke_width = stroke_widths[y, x]
            color = image[y, x].tolist()
            angle = np.deg2rad(135)
            end_x = int(x + stroke_length * np.cos(angle))
            end_y = int(y + stroke_length * np.sin(angle))
            cv2.line(output_image, (x, y), (end_x, end_y), color, thickness=stroke_width)
    return output_image

def edge_detection(image, threshold=50):
    """
    Detect edges in the image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    _, edge_pixels = cv2.threshold(grad_magnitude, threshold, 255, cv2.THRESH_BINARY)
    return edge_pixels.astype(np.uint8)

def clip_strokes(image, edge_pixels):
    """
    Clip strokes at the edges.
    """
    h, w = image.shape[:2]
    output_image = image.copy()
    for y in range(h):
        for x in range(w):
            if edge_pixels[y, x] > 0:
                output_image[y, x] = image[y, x]
    return output_image

def remove_object(image, mask):
    """
    Remove an object from the image using a mask.
    """
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def create_panorama():
    """
    Create a panorama by stitching multiple images.
    """
    layout = [
        [sg.Text('Select images for panorama')],
        # [sg.Input(key='-FILES-'),
        #  sg.FilesBrowse(file_types=(("Image Files", "*.png;*.jpg;*.jpeg"),("All Files", "*.*"),))],
        [sg.Input(key='-FILES-'),
         sg.FilesBrowse(file_types=(("All Files", "*.*"),))],
        [sg.Button('Create Panorama'), sg.Button('Cancel')]
    ]
    window = sg.Window('Panorama Creation', layout)
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, 'Cancel'):
            window.close()
            break
        elif event == 'Create Panorama':
            files = values['-FILES-']
            if files:
                file_list = files.split(';')
                images = []
                for f in file_list:
                    if os.path.isfile(f):
                        img = cv2.imread(f)
                        if img is not None:
                            images.append(img)
                        else:
                            sg.popup(f'Failed to load image: {f}')
                    else:
                        sg.popup(f'File does not exist: {f}')
                if images:
                    if len(images) < 2:
                        sg.popup('Need at least two images to create a panorama.')
                    else:
                        # Initialize the stitcher
                        try:
                            # For OpenCV versions >= 4.0
                            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
                        except AttributeError:
                            # For OpenCV versions < 4.0
                            stitcher = cv2.createStitcher(cv2.Stitcher_PANORAMA)
                        status, pano = stitcher.stitch(images)
                        if status == cv2.Stitcher_OK:
                            # Display the panorama in a new window
                            pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
                            pano_image = np_im_to_data(pano_rgb)
                            pano_layout = [
                                [sg.Image(data=pano_image)],
                                [sg.Button('Save Panorama'), sg.Button('Close')]
                            ]
                            pano_window = sg.Window('Panorama Result', pano_layout)
                            while True:
                                pano_event, pano_values = pano_window.read()
                                if pano_event in (sg.WINDOW_CLOSED, 'Close'):
                                    pano_window.close()
                                    break
                                elif pano_event == 'Save Panorama':
                                    save_filename = sg.popup_get_file('Save Panorama', save_as=True, file_types=(("PNG Files", "*.png"),), default_extension='.png')
                                    if save_filename:
                                        cv2.imwrite(save_filename, pano)
                                        sg.popup('Panorama saved successfully.')
                        else:
                            error_messages = {
                                cv2.Stitcher_ERR_NEED_MORE_IMGS: 'Need more images to create a panorama.',
                                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: 'Homography estimation failed.',
                                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: 'Camera parameters adjustment failed.'
                            }
                            error_message = error_messages.get(status, 'Unknown error occurred during stitching.')
                            sg.popup(f'Error during stitching: {error_message}')
                else:
                    sg.popup('No valid images were loaded.')
            else:
                sg.popup('No files selected.')
            window.close()
            break


def resize_image(image):
    """
    Resize the image with different interpolation methods and aspect ratio option.
    """
    aspect_ratio = image.shape[1] / image.shape[0]

    layout = [
        [sg.Text('Width:'), sg.InputText(str(image.shape[1]), key='-WIDTH-', size=(10,1)),
         sg.Text('Height:'), sg.InputText(str(image.shape[0]), key='-HEIGHT-', size=(10,1))],
        [sg.Checkbox('Maintain Aspect Ratio', default=True, key='-ASPECT-')],
        [sg.Text('Interpolation Method:')],
        [sg.Radio('Nearest Neighbor', 'INTERP', key='-NEAREST-', default=True),
         sg.Radio('Bilinear', 'INTERP', key='-BILINEAR-'),
         sg.Radio('Bicubic', 'INTERP', key='-BICUBIC-')],
        [sg.Button('Resize'), sg.Button('Cancel')]
    ]
    window = sg.Window('Resize Image', layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            window.close()
            return image
        elif event == 'Resize':
            try:
                width = int(values['-WIDTH-'])
                height = int(values['-HEIGHT-'])
                aspect = values['-ASPECT-']
                if aspect:
                    height = int(width / aspect_ratio)
                    window['-HEIGHT-'].update(str(height))
                if values['-NEAREST-']:
                    interp = cv2.INTER_NEAREST
                elif values['-BILINEAR-']:
                    interp = cv2.INTER_LINEAR
                else:
                    interp = cv2.INTER_CUBIC
                resized_image = cv2.resize(image, (width, height), interpolation=interp)
                window.close()
                return resized_image
            except ValueError:
                sg.popup('Please enter valid dimensions.')
        elif event == '-WIDTH-':
            if values['-ASPECT-']:
                try:
                    width = int(values['-WIDTH-'])
                    height = int(width / aspect_ratio)
                    window['-HEIGHT-'].update(str(height))
                except ValueError:
                    pass
        elif event == '-HEIGHT-':
            if values['-ASPECT-']:
                try:
                    height = int(values['-HEIGHT-'])
                    width = int(height * aspect_ratio)
                    window['-WIDTH-'].update(str(width))
                except ValueError:
                    pass
    return image

def display_image(np_image):
    """
    Main function to display the image and handle user interactions.
    """
    original_image = np_image.copy()
    edited_image = np_image.copy()
    history = [edited_image.copy()]
    selection_mask = None

    image_data = np_im_to_data(edited_image)

    height, width, _ = np_image.shape

    # Define the graph element
    graph = sg.Graph(
        canvas_size=(width, height),
        graph_bottom_left=(0, height),
        graph_top_right=(width, 0),
        key='-IMAGE-',
        background_color='white',
        enable_events=True,
        drag_submits=True,
        motion_events=True
    )

    # Adjusted the layout to make all options in 'Selection Tools' visible
    layout = [
        [sg.Column([
            [graph]
        ], element_justification='center'),
        sg.VSeparator(),
        sg.Column([
            [sg.Frame('Adjustments', [
                [sg.Text('Brightness')],
                [sg.Slider(range=(-100,100), default_value=0, orientation='h', size=(20,20), key='-BRIGHTNESS-')],
                [sg.Text('Contrast')],
                [sg.Slider(range=(-100,100), default_value=0, orientation='h', size=(20,20), key='-CONTRAST-')],
                [sg.Text('Sharpen Amount')],
                [sg.Slider(range=(1,100), default_value=1, orientation='h', size=(20,20), key='-SHARPEN-')],
                [sg.Text('Blur Amount')],
                [sg.Slider(range=(0,10), default_value=0, orientation='h', size=(20,20), key='-BLUR-')],
                [sg.Text('Color Temperature')],
                [sg.Slider(range=(-100, 100), default_value=0, orientation='h', size=(20,20), key='-TEMP-')],
                [sg.Button('Apply Adjustments', size=(20,1))]
            ], pad=(10,10))],
            [sg.Frame('Filters', [
                [sg.Text('Select Filter')],
                [sg.Combo(['None', 'Sepia', 'Black and White', 'Oil Painting', 'Oil Painting 2', 'Film Effect'], default_value='None', key='-FILTER-')],
                [sg.Button('Apply Filter', size=(20,1))]
            ], pad=(10,10))],
            [sg.Frame('Selection Tools', [
                [sg.Text('Selection Shape:')],
                [sg.Radio('Rectangle', 'SHAPE', default=True, key='-RECT-')],
                [sg.Radio('Circle', 'SHAPE', key='-CIRCLE-')],
                [sg.Radio('Freehand', 'SHAPE', key='-FREEHAND-')],
                [sg.Text('Brush Size')],
                [sg.Slider(range=(1,20), default_value=5, orientation='h', size=(20,20), key='-BRUSH-')],
                [sg.Button('Clear Selection', size=(20,1))]
            ], pad=(10,10))],
            [sg.Frame('Other Functions', [
                [sg.Button('Object Removal', size=(20,1))],
                [sg.Button('Resize Image', size=(20,1))],
                [sg.Button('Panorama', size=(20,1))],
                [sg.Button('Save Image', size=(20,1))],
                [sg.Button('Undo', size=(20,1))]
            ], pad=(10,10))],
            [sg.Button('Reset Image', size=(20,1))],
            [sg.Button('Exit', size=(20,1))]
        ], element_justification='left', scrollable=True, vertical_scroll_only=True, size=(300, 600))]
    ]

    window = sg.Window('Image Editor', layout, finalize=True)

    # Draw the initial image at the correct location
    graph.draw_image(data=image_data, location=(0, 0))

    drawing = False
    last_point = None

    # Event loop
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        elif event == '-IMAGE-':
            x, y = values['-IMAGE-']
            if x is not None and y is not None:
                if values['-RECT-'] or values['-CIRCLE-']:
                    if not drawing:
                        drawing = True
                        start_point = (x, y)
                        selection_mask = np.zeros((height, width), np.uint8)
                    else:
                        end_point = (x, y)
                        # Create selection mask
                        selection_mask = np.zeros((height, width), np.uint8)
                        if values['-RECT-']:
                            cv2.rectangle(selection_mask, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), 255, -1)
                        elif values['-CIRCLE-']:
                            center = ((start_point[0]+end_point[0])//2, (start_point[1]+end_point[1])//2)
                            radius = int(np.hypot(end_point[0]-start_point[0], end_point[1]-start_point[1])//2)
                            cv2.circle(selection_mask, (int(center[0]), int(center[1])), radius, 255, -1)
                        # Update the selection overlay
                        overlay = edited_image.copy()
                        sky_blue = np.array([135, 206, 235], dtype=np.uint8)
                        alpha = 0.8
                        overlay[selection_mask == 255] = (overlay[selection_mask == 255] * (1 - alpha) + sky_blue * alpha).astype(np.uint8)
                        overlay_data = np_im_to_data(overlay)
                        graph.erase()
                        graph.draw_image(data=overlay_data, location=(0, 0))
                elif values['-FREEHAND-']:
                    if not drawing:
                        drawing = True
                        last_point = (x, y)
                        if selection_mask is None:
                            selection_mask = np.zeros((height, width), np.uint8)
                    else:
                        brush_size = int(values['-BRUSH-'])
                        cv2.line(selection_mask, (int(last_point[0]), int(last_point[1])), (int(x), int(y)), 255, thickness=brush_size)
                        last_point = (x, y)
                        # Update the selection overlay
                        overlay = edited_image.copy()
                        sky_blue = np.array([135, 206, 235], dtype=np.uint8)
                        alpha = 0.8
                        overlay[selection_mask == 255] = (overlay[selection_mask == 255] * (1 - alpha) + sky_blue * alpha).astype(np.uint8)
                        overlay_data = np_im_to_data(overlay)
                        graph.erase()
                        graph.draw_image(data=overlay_data, location=(0, 0))
        elif event.endswith('+UP'):
            if drawing:
                drawing = False
                last_point = None
                # Display final selection overlay
                if selection_mask is not None:
                    overlay = edited_image.copy()
                    sky_blue = np.array([135, 206, 235], dtype=np.uint8)
                    alpha = 0.8
                    overlay[selection_mask == 255] = (overlay[selection_mask == 255] * (1 - alpha) + sky_blue * alpha).astype(np.uint8)
                    overlay_data = np_im_to_data(overlay)
                    graph.erase()
                    graph.draw_image(data=overlay_data, location=(0, 0))
        elif event == 'Apply Adjustments':
            # Apply adjustments to the selected area or the whole image
            if selection_mask is not None:
                # Apply adjustments only to the selected area
                adjusted_area = original_image.copy()
                brightness = values['-BRIGHTNESS-']
                contrast = values['-CONTRAST-']
                sharpen_amount = values['-SHARPEN-']
                blur_amount = values['-BLUR-']
                temp = values['-TEMP-']
                # Extract the selected area
                selected_region = adjusted_area.copy()
                selected_region[selection_mask == 0] = 0
                # Apply adjustments
                if brightness != 0 or contrast != 0:
                    selected_region = adjust_brightness_contrast(selected_region, brightness, contrast)
                if sharpen_amount > 1:
                    selected_region = sharpen_image(selected_region, amount=sharpen_amount)
                if blur_amount > 0:
                    selected_region = blur_image(selected_region, amount=blur_amount)
                if temp != 0:
                    selected_region = adjust_color_temperature(selected_region, temp)
                # Combine the adjusted selected area with the rest of the image
                adjusted_area[selection_mask != 0] = selected_region[selection_mask != 0]
                edited_image = adjusted_area
            else:
                # Apply adjustments to the whole image
                edited_image = original_image.copy()
                brightness = values['-BRIGHTNESS-']
                contrast = values['-CONTRAST-']
                sharpen_amount = values['-SHARPEN-']
                blur_amount = values['-BLUR-']
                temp = values['-TEMP-']
                # Apply adjustments
                if brightness != 0 or contrast != 0:
                    edited_image = adjust_brightness_contrast(edited_image, brightness, contrast)
                if sharpen_amount > 1:
                    edited_image = sharpen_image(edited_image, amount=sharpen_amount)
                if blur_amount > 0:
                    edited_image = blur_image(edited_image, amount=blur_amount)
                if temp != 0:
                    edited_image = adjust_color_temperature(edited_image, temp)
            image_data = np_im_to_data(edited_image)
            graph.erase()
            graph.draw_image(data=image_data, location=(0, 0))
            # Save current state for undo
            history.append(edited_image.copy())
            # Reset selection
            selection_mask = None
        elif event == 'Apply Filter':
            # Apply filter to the selected area or the whole image
            if selection_mask is not None:
                # Apply filter only to the selected area
                filtered_area = original_image.copy()
                filter_type = values['-FILTER-']
                selected_region = filtered_area.copy()
                selected_region[selection_mask == 0] = 0
                # Apply filter
                selected_region = apply_filter(selected_region, filter_type)
                # Combine the filtered selected area with the rest of the image
                filtered_area[selection_mask != 0] = selected_region[selection_mask != 0]
                edited_image = filtered_area
            else:
                # Apply filter to the whole image
                edited_image = original_image.copy()
                filter_type = values['-FILTER-']
                edited_image = apply_filter(edited_image, filter_type)
            image_data = np_im_to_data(edited_image)
            graph.erase()
            graph.draw_image(data=image_data, location=(0, 0))
            # Save current state for undo
            history.append(edited_image.copy())
            # Reset selection
            selection_mask = None
        elif event == 'Object Removal':
            if selection_mask is not None:
                # Apply object removal to the last image in history
                edited_image = history[-1].copy()
                # Save current state for undo
                history.append(edited_image.copy())
                edited_image = remove_object(edited_image, selection_mask)
                image_data = np_im_to_data(edited_image)
                graph.erase()
                graph.draw_image(data=image_data, location=(0, 0))
                # Reset selection mask
                selection_mask = None
            else:
                sg.popup('Please select an area to remove.')
        elif event == 'Clear Selection':
            selection_mask = None
            graph.erase()
            image_data = np_im_to_data(edited_image)
            graph.draw_image(data=image_data, location=(0, 0))
        elif event == 'Resize Image':
            resized_image = resize_image(edited_image)
            if resized_image is not None:
                edited_image = resized_image
                # Reset original image and history with resized image
                original_image = resized_image.copy()
                history = [edited_image.copy()]
                image_data = np_im_to_data(edited_image)
                # Update graph size
                height, width, _ = edited_image.shape
                graph.change_coordinates((0, height), (width, 0))
                graph.Widget.config(width=width, height=height)
                graph.erase()
                graph.draw_image(data=image_data, location=(0, 0))
        elif event == 'Panorama':
            create_panorama()
        elif event == 'Reset Image':
            # Reset to the original image
            edited_image = original_image.copy()
            history = [edited_image.copy()]
            image_data = np_im_to_data(edited_image)
            # Reset graph size to match original image size
            height, width, _ = edited_image.shape
            graph.change_coordinates((0, height), (width, 0))
            graph.Widget.config(width=width, height=height)
            graph.erase()
            graph.draw_image(data=image_data, location=(0, 0))
            # Reset sliders and selection
            window['-BRIGHTNESS-'].update(0)
            window['-CONTRAST-'].update(0)
            window['-SHARPEN-'].update(1)
            window['-BLUR-'].update(0)
            window['-TEMP-'].update(0)
            window['-FILTER-'].update('None')
            selection_mask = None
        elif event == 'Save Image':
            filename = sg.popup_get_file('Save Image', save_as=True, file_types=(("PNG Files", "*.png"),), default_extension='.png')
            if filename:
                cv2.imwrite(filename, cv2.cvtColor(edited_image, cv2.COLOR_RGB2BGR))
                sg.popup('Image saved successfully.')
        elif event == 'Undo':
            if len(history) > 1:
                history.pop()
                edited_image = history[-1].copy()
                image_data = np_im_to_data(edited_image)
                graph.erase()
                graph.draw_image(data=image_data, location=(0, 0))
                # Reset selection
                selection_mask = None
            else:
                sg.popup('No more actions to undo.')
    window.close()

def main():
    """
    Main entry point of the program.
    """
    parser = argparse.ArgumentParser(description='A simple image editor.')
    parser.add_argument('file', nargs='?', help='Image file.')
    args = parser.parse_args()

    if args.file:
        # Load the image using OpenCV
        image = cv2.imread(args.file)
        if image is None:
            sg.popup('Failed to load image.')
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Store the original image
        original_image = image.copy()
        # Resize the image if it's too large
        if image.shape[1] > 800 or image.shape[0] > 600:
            image = cv2.resize(image, (800,600), interpolation=cv2.INTER_LINEAR)
    else:
        filename = sg.popup_get_file('Select an image file', file_types=(("Image Files", "*.png;*.jpg;*.jpeg"),))
        if filename:
            image = cv2.imread(filename)
            if image is None:
                sg.popup('Failed to load image.')
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Store the original image
            original_image = image.copy()
            if image.shape[1] > 800 or image.shape[0] > 600:
                image = cv2.resize(image, (800,600), interpolation=cv2.INTER_LINEAR)
        else:
            sg.popup('No file selected.')
            return

    display_image(image)

if __name__ == '__main__':
    main()
