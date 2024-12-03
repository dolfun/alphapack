class Info:
  image_length = 16
  image_width = 16
  nr_channels = 1
  action_count = 32

def unflatten(data):
  image_shape = (Info.nr_channels, Info.image_length, Info.image_width)

  image_data_size = image_shape[0] * image_shape[1] * image_shape[2]
  image_data = data[:image_data_size]
  image_data = image_data.reshape(image_shape)
  packages_data = data[image_data_size:]

  return image_data, packages_data