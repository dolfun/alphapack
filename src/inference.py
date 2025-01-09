from container_solver import Container
from fastapi import Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from policy_value_network import PolicyValueNetwork
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != 'cuda':
  try:
    import torch_directml
    device = torch_directml.device()
  except ImportError:
    pass

model_path = './temp/policy_value_network.pth'
model = PolicyValueNetwork().to(device)
model.load_state_dict(torch.load(model_path, weights_only=False))
model.eval()

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get('/info')
async def info():
  info = {
    'length': Container.length,
    'height': Container.height,
    'nrPackages': Container.package_count
  }
  return info

@app.post('/infer')
async def infer(request: Request):
  data = await request.json()
  image_data = torch.tensor(data['height_map'], dtype=torch.float32, device=device)
  image_data /= Container.height
  image_data = image_data.view((1, 1, Container.length, Container.length))

  additional_data = []
  for package in data['packages']:
    if package['isPlaced']:
      additional_data.extend([0, 0, 0, 0])
    else:
      l = package['length'] / Container.length
      w = package['width'] / Container.length
      h = package['height'] / Container.height
      additional_data.append(l)
      additional_data.append(w)
      additional_data.append(h)
      additional_data.append(l * w * h)

  with torch.no_grad():
    additional_data = torch.tensor(additional_data, dtype=torch.float32, device=device)
    additional_data = torch.unsqueeze(additional_data, dim=0)

    priors, value = model(image_data, additional_data)
    priors = torch.softmax(torch.squeeze(priors).flatten(), dim=1)
    value = torch.squeeze(value)

  data = {
    'priors': priors.tolist(),
    'value': value.item()
  }

  return data