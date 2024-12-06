from celery import Celery
from policy_value_network import PolicyValueNetwork
import torch

app = Celery(
  'tasks',
  broker='redis://127.0.0.1:6379/0',
  backend='redis://127.0.0.1:6379/0',
)

app.conf.update(
  task_serializer='pickle',
  result_serializer='pickle',
  accept_content=['pickle']
)

policy_value_network = PolicyValueNetwork()

@app.task
def load_model(model_bytes):
  policy_value_network.load_state_dict(torch.load(model_bytes, weights_only=True))
  return 'success'

@app.task
def policy_value_inference(image_data, packages_data):
  with torch.no_grad():
    image_data = torch.tensor(image_data)
    packages_data = torch.tensor(packages_data)

    policy, value = policy_value_network.forward(image_data, packages_data)
    policy = torch.softmax(policy, dim=1)

    result = torch.cat((policy, value), dim=1)
    return result.numpy().tobytes()