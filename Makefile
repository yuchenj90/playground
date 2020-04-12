.PHONY: clean purge lab tunnel ext env

env:
	virtualenv -p python3 env
	env/bin/pip install --upgrade pip
	env/bin/pip install -r requirements.txt
	env/bin/pip install -e .
	make test  # fail early
	make ext

ext:
	# https://code.uberinternal.com/D3842901
	bash -l -c 'source ~/.nvm/nvm.sh; nvm install 12; nvm exec 12 env/bin/jupyter labextension install @jupyter-widgets/jupyterlab-manager @deck.gl/jupyter-widget nbdime-jupyterlab;'

lab:
	bash -l -c 'source ~/.nvm/nvm.sh; nvm exec 12 env/bin/jupyter lab'

clean:
	find . -name '*.pyc' | xargs rm -r
	find . -name '*.ipynb_checkpoints' | xargs rm -r
	find . -name '__pycache__' | xargs rm -r
	find . -name '.pytest_cache' | xargs rm -r
	find . -name '*.egg-info' | xargs rm -r

purge: clean
	rm -rf env

tunnel:
	cerberus -s gairos,queryrunner,querybuilder,wonkamaster,query-result -t queryrunner --no-status-page

test:
	# turn off Deprecation warnings from `imp` and `tornado`
	env/bin/pytest -p no:warnings
