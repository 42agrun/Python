#   Local server for the "Delivery Order Price Calculator"
#   Code by Adrian Grun (egrun@gmx.de)  January 2025
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import delivery_calculator

_RED = "\033[1;31m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[1;33m"
_GREY = "\033[0;37m"

def read_input(string, i):
    input = ""
    while i < len(string) and string[i] != '&':
        input += string[i]
        i += 1
    return input

#   Only 4 attributes are allowed, neither more nor less.
#   The number of ampersands ("&") must be exactly 3.
def parse_input(args):
    data = []
    i = args.find("venue_slug=home-assignment-venue-")
    j = args.find("&cart_value=")
    k = args.find("&user_lat=")
    l = args.find("&user_lon=")
    m = args.count('&')
    if i == -1 or j == -1 or k == -1 or l == -1 or m != 3:
        return "bad"
    i += len("venue_slug=home-assignment-venue-")
    j += len("&cart_value=")
    k += len("&user_lat=")
    l += len("&user_lon=")
    data.append(read_input(args, i))
    data.append(read_input(args, j))
    data.append(read_input(args, k))
    data.append(read_input(args, l))
    return data

class ServerHandler(BaseHTTPRequestHandler):
    #   This function will send a JSON file if everything is fine.
    #   Otherwise, a string of collected error-messages describing
    #   the encountered issues will be sent to the user.
    def send_response(self, response):
        #   This should be able to catch favicon-requests.
        if type(response) is int:
            self.send_response_only(200)
            return
        if type(response) is not str:
            buffer, i = "", 0
            while i < len(response):
                buffer += str(response[i])
                i += 1
            response = buffer
            self.send_response(400)
            self.send_header("content-type", "text/html")
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
        print(f"{_YELLOW}Server response:{_GREY}\n{response}\n")
        self.wfile.write(response.encode())

    def do_GET(self):
        input, time = "bad", datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"{_YELLOW}{time} - ", end="")
        print(f"Received request:{_GREY}\n{self.path}")
        splitter = list(self.path.split('/'))
        if splitter[1] == "favicon.ico":
            self.send_response_only(200)
            return
        if len(splitter) > 3:
            if (splitter[1] == "api" and splitter[2] == "v1" and
                    splitter[3].find('delivery-order-price?') == 0):
                    input = parse_input(splitter[3])
        if type(input) is str:
            self.send_response(400)
            self.send_header('content-type', 'text/html')
            self.end_headers()
            bad = ("Bad request! Expected formatting:<br />"
                             "http://localhost:8000/api/v1/delivery-order-price?"
                             "venue_slug=home-assignment-venue-[LOCATION]&cart_value=[VALUE]"
                             "&user_lat=[X_COORD]&user_lon=[Y_COORD]")
            print(f"{_YELLOW}Server response:{_GREY}\n{bad}\n")
            self.wfile.write(bad.encode())
        else:
            response = delivery_calculator.start_dopc(input)
            self.send_response(response)

def run_server():
    server = HTTPServer(('', 8000), ServerHandler)
    print(f"{_GREY}\nServer has been started on {_GREEN}port 8000{_GREY}. Awaiting client request(s).")
    print(f"{_GREEN}Expected formatting:{_GREY} localhost:8000/api/v1/delivery-order-price?"
                             f"venue_slug=home-assignment-venue-{_YELLOW}[LOCATION]{_GREY}&cart_value={_YELLOW}[VALUE]{_GREY}"
                             f"&user_lat={_YELLOW}[X_COORD]{_GREY}&user_lon={_YELLOW}[Y_COORD]{_GREY}\n")
    server.serve_forever()

if __name__ == "__main__":
    run_server()