from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, PlainTextResponse
import worker

import io
import numpy as np
from package_utils import normalize_packages
from container_solver import Container

app = FastAPI()

@app.post('/policy_value_upload')
async def load_model(file: UploadFile = File(...)):
  content = await file.read()
  content = io.BytesIO(content)
  result = worker.load_model.delay(content)
  return PlainTextResponse(content=result.get())

@app.post('/policy_value_inference')
async def root(request: Request):
  data = await request.body()
  
  batch_size = int(request.headers['batch-size'])
  image_data = []
  packages_data = []
  step_size = len(data) // batch_size
  containers = [Container.unserialize(data[i:i+step_size]) for i in range(0, len(data), step_size)]
  for container in containers:
    height_map = np.array(container.height_map, dtype=np.float32) / container.height
    image_data.append(np.expand_dims(height_map, axis=0))
    packages_data.append(normalize_packages(container.packages))

  result = worker.policy_value_inference.delay(image_data, packages_data)
  return Response(content=result.get(), media_type='application/octet-stream')