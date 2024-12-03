import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import server
import numpy as np
from unflatten import unflatten

import torch
from policy_value_network import PolicyValueNetwork

class InferenceServerHandler(server.ServerHandler):
  policy_value_network = PolicyValueNetwork()

  def request_handler(self, request):
    assert(request['type'].decode('ascii') == 'inference')
    data = request['data']

    data = np.frombuffer(request['data'], dtype=np.float32)

    image_data, packages_data = unflatten(data)
    image_data = torch.tensor(image_data).unsqueeze(0)
    packages_data = torch.tensor(packages_data).unsqueeze(0)

    policy = None
    value = None
    InferenceServerHandler.policy_value_network.eval()
    with torch.no_grad():
      policy, value = InferenceServerHandler.policy_value_network.forward(image_data, packages_data)

    policy = policy.squeeze(dim=0)
    value = value.squeeze(dim=0)

    result = torch.cat((policy, value))

    return result.numpy().tobytes()

def run():
  server.run(handler=InferenceServerHandler)

if __name__ == '__main__':
  run()