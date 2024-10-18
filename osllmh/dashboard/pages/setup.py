"""Setup dashboard."""

import json

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
from dash_extensions.enrich import (
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
)

from osllmh import engine
from osllmh.dashboard.components import dash_store
from osllmh.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/", title="osllmh", order=1)

#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Set up layout."""
    return html.Div(
        [
            html.H1("Setup Page"),
            dcc.Store(id="settings-store", storage_type="session", data=None),
            dbc.Card(
                [
                    dbc.CardHeader("Initialize Engine"),
                    dbc.CardBody(
                        [
                            dbc.Button(
                                "Initialize Engine",
                                id="initialize-engine-button",
                                className="m-2",
                            ),
                            dbc.Button(
                                "Update Index",
                                id="update-index-button",
                                className="m-2",
                            ),
                        ],
                    ),
                ],
                className="mb-2",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Current Settings"),
                    dbc.CardBody(
                        [
                            html.Pre(
                                id="current-settings", style={"whiteSpace": "pre-wrap"}
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "Save Settings",
                                                id="save-settings-button",
                                                color="primary",
                                            ),
                                        ],
                                        width="auto",
                                    ),
                                ],
                            ),
                            html.Div(
                                id="settings-output",
                                className="mt-3",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "border": "1px solid #333333",
                                },
                            ),
                        ]
                    ),
                ],
                className="mb-2",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Update Settings"),
                    dbc.CardBody(
                        [
                            dbc.Form(
                                [
                                    html.H4("Project"),
                                    dbc.Row(
                                        [
                                            dbc.Label("Project Name", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="project-name",
                                                    type="text",
                                                    placeholder="Enter a project name",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.H4("LLM"),
                                    dbc.Row(
                                        [
                                            dbc.Label("LLM Model", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="llm-model",
                                                    type="text",
                                                    placeholder="Enter a model",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("LLM Temperature", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="llm-temperature",
                                                    type="number",
                                                    placeholder="Enter a number",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.H4("Embed Model"),
                                    dbc.Row(
                                        [
                                            dbc.Label("Model Name", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="model-name",
                                                    type="text",
                                                    placeholder="Enter a model",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Embed Batch Size", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="embed-batch-size",
                                                    type="number",
                                                    placeholder="Enter a number",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.H4("Text Splitter"),
                                    dbc.Row(
                                        [
                                            dbc.Label("Chunk Size", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="chunk-size",
                                                    type="number",
                                                    placeholder="Enter a number",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Chunk Overlap", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="chunk-overlap",
                                                    type="number",
                                                    placeholder="Enter a number",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.H4("Prompt"),
                                    dbc.Row(
                                        [
                                            dbc.Label("Context Window", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="context-window",
                                                    type="number",
                                                    placeholder="Enter a number",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Num Output", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="num-output",
                                                    type="number",
                                                    placeholder="Enter a number",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Response Mode", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="response-mode",
                                                    type="text",
                                                    placeholder="Enter a string",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Label("Prompt Section", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="prompt-section",
                                                    type="text",
                                                    placeholder="Enter a string",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.H4("Engine"),
                                    dbc.Row(
                                        [
                                            dbc.Label("Nodes", width=2),
                                            dbc.Col(
                                                dbc.Input(
                                                    id="nodes-similar",
                                                    type="number",
                                                    placeholder="Enter a number",
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Button(
                                                        "Update Settings",
                                                        id="update-settings-button",
                                                        color="primary",
                                                    ),
                                                ],
                                                width="auto",
                                            ),
                                        ]
                                    ),
                                    html.Div(id="settings-output", className="mt-3"),
                                ]
                            )
                        ]
                    ),
                ]
            ),
        ],
        className="container",
    )


#    ____      _ _ _                _
#   / ___|__ _| | | |__   __ _  ___| | _____
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/


# initialize
@callback(
    Output("engine-store", "data", allow_duplicate=True),
    Input("initialize-engine-button", "n_clicks"),
    prevent_initial_call=True,
)
def initialize_engine(n_clicks):
    """Initialize the engine."""
    if n_clicks is None:
        return dash.no_update
    dash_store.engine_instance = engine.Engine()
    engine_loaded = "loaded"

    return engine_loaded


# update index
@callback(
    Input("update-index-button", "n_clicks"),
    prevent_initial_call=True,
)
def create_index(n_clicks):
    """Update the index."""
    if n_clicks is None:
        return
    dash_store.engine_instance.create_index()

    return


# display current settings
@callback(
    Output("current-settings", "children"),
    Input("settings-store", "data"),
    Input("url", "pathname"),
)
def display_current_settings(settings, pathname):
    """Display current settings."""
    if dash_store.engine_instance is None:
        return dash.no_update
    return json.dumps(settings, indent=4)


# update settings
@callback(
    [Output("settings-store", "data"), Output("settings-output", "children")],
    [
        Input("save-settings-button", "n_clicks"),
        Input("update-settings-button", "n_clicks"),
        Input("engine-store", "data"),
    ],
    [
        State("project-name", "value"),
        State("llm-model", "value"),
        State("llm-temperature", "value"),
        State("model-name", "value"),
        State("embed-batch-size", "value"),
        State("chunk-size", "value"),
        State("chunk-overlap", "value"),
        State("context-window", "value"),
        State("num-output", "value"),
        State("response-mode", "value"),
        State("prompt-section", "value"),
        State("nodes-similar", "value"),
    ],
    prevent_initial_call=True,
)
def update_settings(
    click_save,
    click_update,
    engine_loaded,
    project_name,
    llm_model,
    llm_temperature,
    model_name,
    embed_batch_size,
    chunk_size,
    chunk_overlap,
    context_window,
    num_output,
    response_mode,
    prompt_section,
    nodes_similar,
):
    """Update settings."""
    if click_save is None and click_update is None and engine_loaded is None:
        logger.debug("No settings were updated.")
        return dash.no_update, dash.no_update
    e = dash_store.engine_instance
    if click_save:
        logger.info("Saving settings")
        settings = e.get_settings(save=True)
        return settings, "Settings saved!"
    if click_update:
        logger.info("Updating settings")
        settings = e.get_settings(save=False)
        # loop through all the inputs and update the settings only
        # if the input is not None
        updates = {
            "project.name": project_name,
            "llm.model": llm_model,
            "llm.temperature": llm_temperature,
            "embed_model.model_name": model_name,
            "embed_model.batch_size": embed_batch_size,
            "text_splitter.chunk_size": chunk_size,
            "text_splitter.chunk_overlap": chunk_overlap,
            "prompt_helper.context_window": context_window,
            "prompt_helper.num_output": num_output,
            "prompt_helper.response_mode": response_mode,
            "prompt_helper.prompt_section": prompt_section,
            "engine.nodes_similar": nodes_similar,
        }

        # traversing the settings dict to update the values
        for key, value in updates.items():
            if value is not None:
                keys = key.split(".")
                sub_settings = settings
                for k in keys[:-1]:
                    sub_settings = sub_settings[k]
                sub_settings[keys[-1]] = value

        recreate_index = False
        if project_name is not None:
            recreate_index = True
        e.update_settings(settings, recreate_index=recreate_index)
        return settings, "Settings updated!"
    if engine_loaded:
        logger.info("Loading settings")
        settings = e.get_settings()
        return settings, dash.no_update
