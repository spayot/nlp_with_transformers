.ONESHELL:

SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Create conda env from env.yml and compile and install exact pip packages
setup:
    conda env update --prune -f env.yml
    $(CONDA_ACTIVATE) <<env_name>>
	pip install -e .