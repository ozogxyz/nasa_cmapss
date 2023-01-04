import yaml


def read_params(params_filepath):
  # read hyperparameters
  with open(params_filepath, 'r') as file:
      params = yaml.safe_load(file)

  # dataset params
  fd = params.get('dataset').get('filename')
  batch_size = params.get('dataset').get('batch_size')
  window_size = params.get('dataset').get('window_size')

  # model params
  in_channels = params.get('model').get('in_channels')
  out_channels = params.get('model').get('out_channels')
  kernel_size = params.get('model').get('kernel_size')
  stride = params.get('model').get('stride')
  maxpool_kernel = params.get('model').get('maxpool_kernel')
  num_classes = params.get('model').get('num_classes')
  hidden_size = params.get('model').get('hidden_size')
  num_layers = params.get('model').get('num_layers')
  maxpool_stride = params.get('model').get('maxpool_stride')
  dropout = params.get('model').get('dropout')

  # training params
  max_epochs = params.get('training').get('epochs')
  patience = params.get('training').get('patience')
  min_delta = params.get('training').get('min_delta')
  lr = params.get('training').get('lr')
  
  return fd, batch_size, window_size, in_channels, out_channels, kernel_size, stride, maxpool_kernel, maxpool_stride, num_classes, hidden_size, num_layers, max_epochs, patience, min_delta, lr, dropout