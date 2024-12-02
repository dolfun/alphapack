import server
import numpy as np
from unflatten import Info, unflatten

class InferenceServerHandler(server.ServerHandler):
  def request_handler(self, request):
    assert(request['type'].decode('ascii') == 'inference')
    data = request['data']
    data = np.frombuffer(request['data'], dtype=np.float32)

    image_data, packages_data = unflatten(data)

    result = np.zeros(Info.action_count + 1, dtype=np.float32)
    result[:-1] = 1.0 / Info.action_count
    result[-1] = 0.5

    return result.tobytes()

def run():
  server.run(handler=InferenceServerHandler)

if __name__ == '__main__':
  run()