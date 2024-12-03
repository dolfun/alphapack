from http.server import BaseHTTPRequestHandler, HTTPServer
from requests_toolbelt.multipart import decoder
import re

def parse_multipart(content, content_type):
  multipart_data = decoder.MultipartDecoder(content, content_type)
  result = {}
  for part in multipart_data.parts:
    name = part.headers[b'Content-Disposition'].decode('ascii')
    name = re.search(r'name=["\'](.*?)["\'];?', name).group(1)
    result[name] = part.content
  return result

class ServerHandler(BaseHTTPRequestHandler):
  def do_POST(self):
    content_length = int(self.headers['Content-Length'])
    content_type = self.headers['Content-Type']
    request_content = self.rfile.read(content_length)
    request_content = parse_multipart(request_content, content_type)
    response_in_binary = self.request_handler(request_content)

    self.send_response(200)
    self.send_header('Content-Type', 'application/octet-stream')
    self.send_header('Content-Length', len(response_in_binary))
    self.end_headers()
    self.wfile.write(response_in_binary)

def run(*, server=HTTPServer, handler=ServerHandler, port=8000):
  server_address = ('', port)
  httpd = server(server_address, handler)

  sa = httpd.socket.getsockname()
  print(f'Starting HTTP server on address: {sa[0]}, port: {sa[1]}')

  httpd.serve_forever()