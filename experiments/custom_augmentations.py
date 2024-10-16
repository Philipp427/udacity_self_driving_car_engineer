import tensorflow as tf

def random_cutout(image, max_boxes, min_box_size, max_box_size):
    # Get the height and width of the image
    img_height, img_width, _ = image.shape
    
    # Loop to create multiple cutouts
    for _ in range(max_boxes):
        # Randomly determine the height of the cutout box
        box_height = tf.random.uniform([], min_box_size * img_height, max_box_size * img_height, dtype=tf.int32)
        
        # Randomly determine the width of the cutout box
        box_width = tf.random.uniform([], min_box_size * img_width, max_box_size * img_width, dtype=tf.int32)
        
        # Randomly determine the y-coordinate of the top-left corner of the cutout box
        y = tf.random.uniform([], 0, img_height - box_height, dtype=tf.int32)
        
        # Randomly determine the x-coordinate of the top-left corner of the cutout box
        x = tf.random.uniform([], 0, img_width - box_width, dtype=tf.int32)
        
        # Create the cutout by setting the pixels in the box to zero (black)
        image = tf.tensor_scatter_nd_update(
            image, 
            tf.stack([tf.range(y, y + box_height), tf.range(x, x + box_width)], axis=-1), 
            tf.zeros([box_height, box_width, 3])
        )
    
    # Return the modified image
    return image

