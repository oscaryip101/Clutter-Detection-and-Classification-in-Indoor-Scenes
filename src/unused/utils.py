import cv2
import matplotlib.pyplot as plt

def load_image(path):
    """
    Load an image from disk using OpenCV.

    Parameters
    ----------
    path : str
        File path to the image.

    Returns
    -------
    img : np.ndarray
        Loaded BGR image, or None if loading fails.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img


def save_image(path, img):
    """
    Save an image to disk.

    Parameters
    ----------
    path : str
        Output file path.
    img : np.ndarray
        Image to save.
    """
    cv2.imwrite(path, img)


def draw_boxes(img, boxes, color=(0, 255, 0)):
    """
    Draw bounding boxes (and optional labels) on an image.

    Parameters
    ----------
    img : np.ndarray
        Image on which to draw.
    boxes : list of tuples
        - For differencing boxes: (x, y, w, h)
        - For YOLO boxes with class info: (x, y, w, h, cls_id, conf, cls_name)
    color : tuple
        BGR color for the boxes.

    Returns
    -------
    img_drawn : np.ndarray
        Image with drawn boxes and labels (if available).
    """
    img_drawn = img.copy()

    for box in boxes:
        if len(box) < 4:
            continue

        x, y, w, h = box[0:4]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cv2.rectangle(img_drawn, (x, y), (x + w, y + h), color, 2)

        # 如果包含类别名，就在框上方写文字
        label_text = None
        if len(box) >= 7:
            cls_name = str(box[6])
            conf = box[5]
            try:
                label_text = f"{cls_name} {conf:.2f}"
            except Exception:
                label_text = cls_name

        if label_text is not None:
            # 在框上方画一个背景条，防止文字看不清
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            text_x = x
            text_y = max(0, y - 5)

            cv2.rectangle(
                img_drawn,
                (text_x, text_y - th - baseline),
                (text_x + tw, text_y + baseline),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                img_drawn,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return img_drawn


def show_image(title, img):
    """
    Display an image using matplotlib (handles BGR → RGB conversion).

    Parameters
    ----------
    title : str
        Title for the plot window.
    img : np.ndarray
        Image to display.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
