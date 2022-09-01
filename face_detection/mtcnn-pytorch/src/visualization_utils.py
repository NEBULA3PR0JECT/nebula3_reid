from PIL import ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np


def show_bboxes(img, bounding_boxes, facial_landmarks=[], plot_conf=False):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """
    if plot_conf:
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()


    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline='white')
        if plot_conf:
            (left, right, top, bottom) = (b[0], b[2], b[1], b[3])
            display_str_list = str(b[-1].__format__('.2f'))
            display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
            # Each display_str has a top and bottom margin of 0.05x.
            total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

            if top > total_display_str_height:
                text_bottom = top + total_display_str_height
            else:
                text_bottom = bottom # + total_display_str_height

            text_width, text_height = font.getsize(display_str_list)
            margin = np.ceil(0.05 * text_height)

            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str_list,
                fill='black',
                font=font)
            # print("confidence{}".format(display_str_list))
            text_bottom -= text_height - 2 * margin


    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline='blue')
    return img_copy
