import os
from http.server import BaseHTTPRequestHandler, HTTPServer

class KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Environment is awake and ready.")
        
    def do_POST(self):
        if self.path == '/reset':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "reset_successful"}')
        else:
            self.send_response(404)
            self.end_headers()

def main():
    port = int(os.environ.get("PORT", 7860))
    server = HTTPServer(("0.0.0.0", port), KeepAliveHandler)
    print(f"[INFO] OpenEnv server entrypoint running on port {port}...", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
