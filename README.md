# Osllmh
Simple helper for the llama index project.

## Overview
This is a simple helper for the llama index project.

![overview](docs/screenshots/osllmh_demo.gif)

## Installation

### Local Install
To install, this repository can be installed by running the following command in 
the environment of choice.

The following command can be run to install the packages in the pyproject.toml file.

```
uv pip install -e .
```

### Docker Install
The package can also be run in docker which provides a containerized environment, and can host the web dashboard.

```bash
version: "3.8"
services:
  osllmh:
    container_name: osllmh
    image: dmbymdt/osllmh
    restart: unless-stopped
    command: gunicorn -b 0.0.0.0:8001 osllmh.dashboard.app:server
    environment:
      OSLLMH_INPUTS_PATH: /code/osllmh # setting the inputs path for osllmh
      OPENAI_API_KEY: $OPEN_API_KEY
    ports:
      - $OSLLMH_PORT:8001
    volumes:
      - $DOCKERDIR/osllmh:/code/osllmh # mounting the files directory
```

### Environment Varibles

Envrionment variables should be set to be able to use the package.

- `OSLLMH_INPUTS_PATH`: The path to the directory where the files are stored.
- `OPENAI_API_KEY`: The openai api key for the openai api.

## Usage

### CLI

CLI can be used for easier commands of python scripts for both portfolio or manager. 

```commandline
osllmh dashboard
```

It also can be run locally by going to the root folder and running below.

```python
python -m osllmh.dashboard.app
```

### Common Use Cases

- start an engine class
- run a query
- update the index

```python
from osllmh.engine import Engine

engine = Engine()
response = engine.query("What is the capital of France?")
response.response
engine.create_index()
```

## Other Tools
### Jupyter Lab Usage

To have conda environments work with Jupyter Notebooks a kernel needs to be defined. This can be done defining a kernel, shown below when
in the conda environment.

```
python -m ipykernel install --user --name=osllmh
```
### Logging

If wanting to get more detail in output of messages the logging can increased
```python
from osllmh.utils import config_helper
config_helper.set_log_level("DEBUG")
```