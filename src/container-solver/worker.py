from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, PlainTextResponse

from policy_value_network import PolicyValueNetwork
from package_utils import normalize_packages
from container_solver import Container

import numpy as np
import torch

import io

policy_value_network = PolicyValueNetwork()
app = FastAPI()

@app.post('/policy_value_upload')
async def load_model(file: UploadFile = File(...)):
  content = await file.read()
  content = io.BytesIO(content)
  policy_value_network.load_state_dict(torch.load(content, weights_only=True))
  return PlainTextResponse(content='success')

@app.post('/policy_value_inference')
async def root(request: Request):
  data = await request.body()
  
  container = Container.unserialize(data)
  height_map = np.array(container.height_map, dtype=np.float32) / container.height
  height_map = torch.tensor(height_map).unsqueeze(0).unsqueeze(0)
  packages_data = torch.tensor(normalize_packages(container.packages)).unsqueeze(0)
  
  with torch.no_grad():
    policy, value = policy_value_network.forward(height_map, packages_data)
    policy = policy.squeeze(dim=0)
    policy = torch.softmax(policy, dim=0)
    value = value.squeeze(dim=0)
    result = torch.cat((policy, value))

    return Response(content=result.numpy().tobytes(), media_type='application/octet-stream')