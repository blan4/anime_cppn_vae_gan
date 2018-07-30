server:
	cd tmp/images;python -m http.server 8080

train: venv
	python main.py

venv:
	source venv/bin/activate