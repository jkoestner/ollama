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
from osllmh.dashboard.components import engine_store
from osllmh.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/setup", title="osllmh - Setup", order=1)

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
                                "Initialize Engine", id="initialize-engine-button"
                            ),
                        ],
                    ),
                ],
                className="mb-4",
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
                        ]
                    ),
                ],
                className="mb-4",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Update Settings"),
                    dbc.CardBody(
                        [
                            dbc.Form(
                                [
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
    engine_store.engine_instance = engine.Engine()
    engine_loaded = "loaded"

    return engine_loaded


# display current settings
@callback(
    Output("current-settings", "children"),
    Input("settings-store", "data"),
    Input("url", "pathname"),
)
def display_current_settings(settings, pathname):
    """Display current settings."""
    logger.info(f"engine_loaded: {engine_store.engine_instance}")
    if engine_store.engine_instance is None:
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
        State("llm-model", "value"),
        State("llm-temperature", "value"),
        State("model-name", "value"),
        State("embed-batch-size", "value"),
        State("chunk-size", "value"),
        State("chunk-overlap", "value"),
        State("context-window", "value"),
        State("num-output", "value"),
    ],
    prevent_initial_call=True,
)
def update_settings(
    click_save,
    click_update,
    engine_loaded,
    llm_model,
    llm_temperature,
    model_name,
    embed_batch_size,
    chunk_size,
    chunk_overlap,
    context_window,
    num_output,
):
    """Update settings."""
    if click_save is None and click_update is None and engine_loaded is None:
        logger.debug("No settings were updated.")
        return dash.no_update, dash.no_update
    e = engine_store.engine_instance
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
            "llm.model": llm_model,
            "llm.temperature": llm_temperature,
            "embed_model.model_name": model_name,
            "embed_model.batch_size": embed_batch_size,
            "text_splitter.chunk_size": chunk_size,
            "text_splitter.chunk_overlap": chunk_overlap,
            "prompt_helper.context_window": context_window,
            "prompt_helper.num_output": num_output,
        }

        for key, value in updates.items():
            if value is not None:
                keys = key.split(".")
                sub_settings = settings
                for k in keys[:-1]:
                    sub_settings = sub_settings[k]
                sub_settings[keys[-1]] = value

        e.load_settings(settings)
        return settings, "Settings updated!"
    if engine_loaded:
        logger.info("Loading settings")
        settings = e.get_settings()
        return settings, dash.no_update
