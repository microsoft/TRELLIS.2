import re
import numpy as np
import cv2
import torch
import contextlib


# Dictionary utils
def _dict_merge(dicta, dictb, prefix=''):
    """
    Merge two dictionaries.
    """
    assert isinstance(dicta, dict), 'input must be a dictionary'
    assert isinstance(dictb, dict), 'input must be a dictionary'
    dict_ = {}
    all_keys = set(dicta.keys()).union(set(dictb.keys()))
    for key in all_keys:
        if key in dicta.keys() and key in dictb.keys():
            if isinstance(dicta[key], dict) and isinstance(dictb[key], dict):
                dict_[key] = _dict_merge(dicta[key], dictb[key], prefix=f'{prefix}.{key}')
            else:
                raise ValueError(f'Duplicate key {prefix}.{key} found in both dictionaries. Types: {type(dicta[key])}, {type(dictb[key])}')
        elif key in dicta.keys():
            dict_[key] = dicta[key]
        else:
            dict_[key] = dictb[key]
    return dict_


def dict_merge(dicta, dictb):
    """
    Merge two dictionaries.
    """
    return _dict_merge(dicta, dictb, prefix='')


def dict_foreach(dic, func, special_func={}):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            dic[key] = dict_foreach(dic[key], func)
        else:
            if key in special_func.keys():
                dic[key] = special_func[key](dic[key])
            else:
                dic[key] = func(dic[key])
    return dic


def dict_reduce(dicts, func, special_func={}):
    """
    Reduce a list of dictionaries. Leaf values must be scalars.
    """
    assert isinstance(dicts, list), 'input must be a list of dictionaries'
    assert all([isinstance(d, dict) for d in dicts]), 'input must be a list of dictionaries'
    assert len(dicts) > 0, 'input must be a non-empty list of dictionaries'
    all_keys = set([key for dict_ in dicts for key in dict_.keys()])
    reduced_dict = {}
    for key in all_keys:
        vlist = [dict_[key] for dict_ in dicts if key in dict_.keys()]
        if isinstance(vlist[0], dict):
            reduced_dict[key] = dict_reduce(vlist, func, special_func)
        else:
            if key in special_func.keys():
                reduced_dict[key] = special_func[key](vlist)
            else:
                reduced_dict[key] = func(vlist)
    return reduced_dict


def dict_any(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if dict_any(dic[key], func):
                return True
        else:
            if func(dic[key]):
                return True
    return False


def dict_all(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if not dict_all(dic[key], func):
                return False
        else:
            if not func(dic[key]):
                return False
    return True


def dict_flatten(dic, sep='.'):
    """
    Flatten a nested dictionary into a dictionary with no nested dictionaries.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    flat_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            sub_dict = dict_flatten(dic[key], sep=sep)
            for sub_key in sub_dict.keys():
                flat_dict[str(key) + sep + str(sub_key)] = sub_dict[sub_key]
        else:
            flat_dict[key] = dic[key]
    return flat_dict


# Context utils
@contextlib.contextmanager
def nested_contexts(*contexts):
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield


# Image utils
def make_grid(images, nrow=None, ncol=None, aspect_ratio=None):
    num_images = len(images)
    if nrow is None and ncol is None:
        if aspect_ratio is not None:
            nrow = int(np.round(np.sqrt(num_images / aspect_ratio)))
        else:
            nrow = int(np.sqrt(num_images))
        ncol = (num_images + nrow - 1) // nrow
    elif nrow is None and ncol is not None:
        nrow = (num_images + ncol - 1) // ncol
    elif nrow is not None and ncol is None:
        ncol = (num_images + nrow - 1) // nrow
    else:
        assert nrow * ncol >= num_images, 'nrow * ncol must be greater than or equal to the number of images'
    
    if images[0].ndim == 2:
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1]), dtype=images[0].dtype)
    else:
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1], images[0].shape[2]), dtype=images[0].dtype)
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        grid[row * img.shape[0]:(row + 1) * img.shape[0], col * img.shape[1]:(col + 1) * img.shape[1]] = img
    return grid


def notes_on_image(img, notes=None):
    img = np.pad(img, ((0, 32), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if notes is not None:
        img = cv2.putText(img, notes, (0, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



def text_image(text, resolution=(512, 512), max_size=0.5, h_align="left", v_align="center"):
    """
    Draw text on an image of the given resolution. The text is automatically wrapped
    and scaled so that it fits completely within the image while preserving any explicit
    line breaks and original spacing. Horizontal and vertical alignment can be controlled
    via flags.
    
    Parameters:
        text (str): The input text. Newline characters and spacing are preserved.
        resolution (tuple): The image resolution as (width, height).
        max_size (float): The maximum font size.
        h_align (str): Horizontal alignment. Options: "left", "center", "right".
        v_align (str): Vertical alignment. Options: "top", "center", "bottom".
        
    Returns:
        numpy.ndarray: The resulting image (BGR format) with the text drawn.
    """
    width, height = resolution
    # Create a white background image
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Set margins and compute available drawing area
    margin = 10
    avail_width = width - 2 * margin
    avail_height = height - 2 * margin

    # Choose OpenCV font and text thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    # Ratio for additional spacing between lines (relative to the height of "A")
    line_spacing_ratio = 0.5

    def wrap_line(line, max_width, font, thickness, scale):
        """
        Wrap a single line of text into multiple lines such that each line's
        width (measured at the given scale) does not exceed max_width.
        This function preserves the original spacing by splitting the line into tokens
        (words and whitespace) using a regular expression.
        
        Parameters:
            line (str): The input text line.
            max_width (int): Maximum allowed width in pixels.
            font (int): OpenCV font identifier.
            thickness (int): Text thickness.
            scale (float): The current font scale.
            
        Returns:
            List[str]: A list of wrapped lines.
        """
        # Split the line into tokens (words and whitespace), preserving spacing
        tokens = re.split(r'(\s+)', line)
        if not tokens:
            return ['']
        
        wrapped_lines = []
        current_line = ""
        for token in tokens:
            candidate = current_line + token
            candidate_width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
            if candidate_width <= max_width:
                current_line = candidate
            else:
                # If current_line is empty, the token itself is too wide;
                # break the token character by character.
                if current_line == "":
                    sub_token = ""
                    for char in token:
                        candidate_char = sub_token + char
                        if cv2.getTextSize(candidate_char, font, scale, thickness)[0][0] <= max_width:
                            sub_token = candidate_char
                        else:
                            if sub_token:
                                wrapped_lines.append(sub_token)
                            sub_token = char
                    current_line = sub_token
                else:
                    wrapped_lines.append(current_line)
                    current_line = token
        if current_line:
            wrapped_lines.append(current_line)
        return wrapped_lines

    def compute_text_block(scale):
        """
        Wrap the entire text (splitting at explicit newline characters) using the
        provided scale, and then compute the overall width and height of the text block.
        
        Returns:
            wrapped_lines (List[str]): The list of wrapped lines.
            block_width (int): Maximum width among the wrapped lines.
            block_height (int): Total height of the text block including spacing.
            sizes (List[tuple]): A list of (width, height) for each wrapped line.
            spacing (int): The spacing between lines (computed from the scaled "A" height).
        """
        # Split text by explicit newlines
        input_lines = text.splitlines() if text else ['']
        wrapped_lines = []
        for line in input_lines:
            wrapped = wrap_line(line, avail_width, font, thickness, scale)
            wrapped_lines.extend(wrapped)
            
        sizes = []
        for line in wrapped_lines:
            (text_size, _) = cv2.getTextSize(line, font, scale, thickness)
            sizes.append(text_size)  # (width, height)
            
        block_width = max((w for w, h in sizes), default=0)
        # Use the height of "A" (at the current scale) to compute line spacing
        base_height = cv2.getTextSize("A", font, scale, thickness)[0][1]
        spacing = int(line_spacing_ratio * base_height)
        block_height = sum(h for w, h in sizes) + spacing * (len(sizes) - 1) if sizes else 0
        
        return wrapped_lines, block_width, block_height, sizes, spacing

    # Use binary search to find the maximum scale that allows the text block to fit
    lo = 0.001
    hi = max_size
    eps = 0.001  # convergence threshold
    best_scale = lo
    best_result = None

    while hi - lo > eps:
        mid = (lo + hi) / 2
        wrapped_lines, block_width, block_height, sizes, spacing = compute_text_block(mid)
        # Ensure that both width and height constraints are met
        if block_width <= avail_width and block_height <= avail_height:
            best_scale = mid
            best_result = (wrapped_lines, block_width, block_height, sizes, spacing)
            lo = mid  # try a larger scale
        else:
            hi = mid  # reduce the scale

    if best_result is None:
        best_scale = 0.5
        best_result = compute_text_block(best_scale)
        
    wrapped_lines, block_width, block_height, sizes, spacing = best_result

    # Compute starting y-coordinate based on vertical alignment flag
    if v_align == "top":
        y_top = margin
    elif v_align == "center":
        y_top = margin + (avail_height - block_height) // 2
    elif v_align == "bottom":
        y_top = margin + (avail_height - block_height)
    else:
        y_top = margin + (avail_height - block_height) // 2  # default to center if invalid flag

    # For cv2.putText, the y coordinate represents the text baseline;
    # so for the first line add its height.
    y = y_top + (sizes[0][1] if sizes else 0)

    # Draw each line with horizontal alignment based on the flag
    for i, line in enumerate(wrapped_lines):
        line_width, line_height = sizes[i]
        if h_align == "left":
            x = margin
        elif h_align == "center":
            x = margin + (avail_width - line_width) // 2
        elif h_align == "right":
            x = margin + (avail_width - line_width)
        else:
            x = margin  # default to left if invalid flag

        cv2.putText(img, line, (x, y), font, best_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += line_height + spacing

    return img


def save_image_with_notes(img, path, notes=None):
    """
    Save an image with notes.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = notes_on_image(img, notes)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# debug utils

def atol(x, y):
    """
    Absolute tolerance.
    """
    return torch.abs(x - y)


def rtol(x, y):
    """
    Relative tolerance.
    """
    return torch.abs(x - y) / torch.clamp_min(torch.maximum(torch.abs(x), torch.abs(y)), 1e-12)


# print utils
def indent(s, n=4):
    """
    Indent a string.
    """
    lines = s.split('\n')
    for i in range(1, len(lines)):
        lines[i] = ' ' * n + lines[i]
    return '\n'.join(lines)

