.PHONY: clean purge lab tunnel ext env

env:
	virtualenv -p python3.8 multitask-env
	multitask-env/bin/pip install --upgrade pip
	multitask-env/bin/pip install -r requirements.txt

lab:
	env/bin/jupyter lab
