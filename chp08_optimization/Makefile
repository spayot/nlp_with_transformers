.ONESHELL:

SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Create conda env from env.yml and compile and install exact pip packages
setup: tfm8.yml
	conda env update --prune -f tfm8.yml
	$(CONDA_ACTIVATE) tfm8
	pip install -e .