.ONESHELL:

SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Create conda env from env.yml and compile and install exact pip packages
setup: tfm9.yml
	conda env update --prune -f tfm9.yml
	$(CONDA_ACTIVATE) tfm9
	pip install -e .