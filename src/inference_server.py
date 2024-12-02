import server
import numpy as np

class InferenceServerHandler(server.ServerHandler):
  def request_handler(self, request):
    assert(request['type'].decode('ascii') == 'inference')
    data = request['data']
    data = np.frombuffer(request['data'], dtype=np.float32).copy()
    data *= 2.0
    return data.tobytes()

def run():
  server.run(handler=InferenceServerHandler)

if __name__ == '__main__':
  run()