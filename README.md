# collective-chant
A website to facilitate a collective chant or prayer. Under development.

## Installation
During development I am running the Collective Chant software on a local Linux box, inside a Python virtual environment. The software uses the Python FastAPI module to serve web pages over my local network to a browser running on my Windows system. I edited my Windows hosts file so that I can use the URL https://fastapi.local to open the main page. To use HTTPS, certificate files were created and placed in the project directory, but a port other than 443 could have been used for testing so that a certificate would not be needed.

1. Install Python
2. Create a project directory cc: mkdir cc
3. Inside cc, create a virtual environment: python3 -m venv .venv
4. Activate the venv: source .venv/bin/activate
5. Download the project software from this Github repository (Download ZIP, under the Code button) and place it in the cc directory.
6. Install dependencies: pip install -r requirements.txt
7. If using HTTPS, run: sudo setcap 'cap_net_bind_service=+ep' $(readlink -f .venv/bin/python3) to allow binding to port 443.
8. If using HTTPS, create a certificate using mkcert, for example, and place fastapi.crt, fastapi.key, fastapi.pfx in the project directory.
9. Start the software: python3 main.py
10. Open the main Collective Chant page in a browser after editing the hosts file: https://fastapi.local

Of course it is possible for the server and browser to be on the same system. I used a local Linux box for the server to approximate using a public server.
