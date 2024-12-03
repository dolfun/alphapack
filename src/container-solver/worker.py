from fastapi import FastAPI, Request
from fastapi.responses import Response

from policy_value_network import PolicyValueNetwork
from container_solver import Container
from package_utils import normalize_packages

import numpy as np
import torch

policy_value_network = PolicyValueNetwork()
policy_value_network.eval()

app = FastAPI()

@app.post('/policy_value_inference')
async def root(request: Request):
  data = await request.body()
  
  container = Container.unserialize(data)
  image_data = np.array(container.height_map, dtype=np.float32) / container.height
  image_data = torch.tensor(image_data).unsqueeze(0).unsqueeze(0)
  packages_data = torch.tensor(normalize_packages(container.packages)).unsqueeze(0)
  
  with torch.no_grad():
    policy, value = policy_value_network.forward(image_data, packages_data)

    policy = policy.squeeze(dim=0)
    value = value.squeeze(dim=0)
    result = torch.cat((policy, value))

    return Response(content=result.numpy().tobytes(), media_type='application/octet-stream')