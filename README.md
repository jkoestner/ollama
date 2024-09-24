# Osllmh
Simple helper for the llama index project.

## Overview
This is a simple helper for the llama index project.

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

It also can be run locally by going to the dashboard folder and running below.

```python
python app.py
```