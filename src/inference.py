from bin_packing_solver import State
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
    'length': State.bin_length,
    'height': State.bin_height,
    'itemCount': State.item_count
  }
  return info

@app.post('/infer')
async def infer(request: Request):
  data = await request.json()
  image_data = torch.tensor(data['height_map'], dtype=torch.float32, device=device)
  image_data /= State.bin_height
  image_data = image_data.view((State.bin_length, State.bin_length))

  feasibility_mask = torch.tensor(data['mask'], dtype=torch.float32, device=device)
  feasibility_mask = feasibility_mask.view((State.bin_length, State.bin_length))

  image_data = torch.stack((image_data, feasibility_mask))
  image_data = image_data.view((1, 2, State.bin_length, State.bin_length))

  additional_data = []
  for item in data['items']:
    if item['isPlaced']:
      additional_data.extend([0, 0, 0, 0])
    else:
      l = item['length'] / State.bin_length
      w = item['width'] / State.bin_length
      h = item['height'] / State.bin_height
      additional_data.append(l)
      additional_data.append(w)
      additional_data.append(h)
      additional_data.append(l * w * h)

  with torch.no_grad():
    additional_data = torch.tensor(additional_data, dtype=torch.float32, device=device)
    additional_data = torch.unsqueeze(additional_data, dim=0)

    priors, value = model(image_data, additional_data)
    priors = torch.softmax(torch.squeeze(priors).flatten(), dim=0)
    value = torch.squeeze(value)

  data = {
    'priors': priors.tolist(),
    'value': value.item()
  }

  return data