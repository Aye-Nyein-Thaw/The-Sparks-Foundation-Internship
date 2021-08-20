def xywh2xyxy(x):
  """
  Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format

  arguments:
    x(numpy array): bounding boxes in yolo format
                    (x_center, y_center, width, height)

  returns:
    y(numpy array): bounding boxes in [x1, y1, x2, y2] format
                    x1, y1 = top left
                    x2, y2 = lower right
  """
  
  y = np.copy(x)
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

  return y


def resize_bboxes(bboxes, img_height, img_width):
  """
  Resize bounding boxes according to image height and width
  """

  y = np.copy(bboxes)
  y[:,0] = y[:,0] * img_width
  y[:,1] = y[:,1] * img_height
  y[:,2] = y[:,2] * img_width
  y[:,3] = y[:,3] * img_height

  return y


def get_image_with_bbox(image_dir, label_file_dir, color_map, text_map):
  """
  Read image and label file.
  Plot bounding boxes and text on the image and returns it. 

  arguments:
    image_dir(str): path to image file
    label_file_dir(str): path to label txt file
    color_map(dict): a dictionary mapping classes and colors
    text_map(dict): a dictionary mapping classes and label text

  returns:
    img_with_bbox(numpy array): image with bounding box(es) and text plotted on it
  """

  # read image file
  img = np.array(Image.open(image_dir))
  img_h, img_w = img.shape[0], img.shape[1]

  # read label file
  with open(label_file_dir) as file:
    lines = file.readlines()
  
  lines = [l.strip().split() for l in lines]
  lines = np.array(lines).astype(np.float)

  # get bounding boxes in yolov5 format
  bboxes = lines[:, 1:]

  # get scores for bounding boxes
  scores = lines[:,0]

  # convert from (x_center, y_center, w, h) to (upper left, lower right)
  bboxes = xywh2xyxy(bboxes)

  # scale bboxes to original image size
  bboxes = resize_bboxes(bboxes, img_h, img_w)

  thickness = 3  #in pixels
 
  for box, score in zip(bboxes, scores):
    start_point = (int(box[0]), int(box[1])) # x,y - top left corner
    end_point = (int(box[2]), int(box[3])) # x,y - bottom right corner
    
    # draw one bbox on image at a time
    img_with_bbox = cv2.rectangle(img, start_point, end_point, 
                                  color_map[score], thickness)
    
    # put rectangle for text above bbox
    (w, h), _ = cv2.getTextSize(text_map[score], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    img_with_bbox = cv2.rectangle(img_with_bbox, 
                                  (start_point[0], start_point[1] - 30), 
                                  (start_point[0] + w, start_point[1]), 
                                  color_map[score], -1)
    # put text above bbox
    img_with_bbox = cv2.putText(img_with_bbox, 
                                text_map[score], 
                                (start_point[0],start_point[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

  return img_with_bbox


def plot_sample_images(dataset_dir, color_map, text_map,  split = 'val', rows = 2, columns = 3):
  """
  Displays multiple random images from training or validation set in one figure

  arguments:
    dataset_dir(str): path to dataset for YOLOv5
    color_map(dict): a dictionary mapping classes and colors
    text_map(dict): a dictionary mapping classes and label text
    split(str): 'train' or 'val'. Split from which random images will be displayed
    rows(int): total number of rows
    columns(int): total number of columns
  """

  img_dir_list = sorted(glob.glob(f'{dataset_dir}/{split}/images/*'))
  label_dir_list = sorted(glob.glob(f'{dataset_dir}/{split}/labels/*'))

  # create figure
  fig = plt.figure(figsize=(15, 10))

  # total images to display 
  total = rows * columns

  # get indices of random images in all image list
  random_img_indices = random.sample(range(len(img_dir_list)), total)

  for i in range(total):
    idx = random_img_indices[i] # index of image
    img_dir = img_dir_list[idx] # image directory
    label_dir = label_dir_list[idx] # label directory

    # get image with boxes plotted
    image_with_bboxes = get_image_with_bbox(img_dir, label_dir, color_map, text_map)
    
    fig.add_subplot(rows, columns, i+1)

    # showing image
    plt.imshow(image_with_bboxes)
    plt.axis('off')
    plt.title(f"Sample {i+1}")
