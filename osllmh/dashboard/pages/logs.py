"""Logs dashboard."""

import os

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
from dash_extensions.enrich import (
    Input,
    Output,
    callback,
    html,
)

from osllmh.dashboard.components import dash_store
from osllmh.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/logs", title="osllmh - Logs", order=3)

#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Log layout."""
    return html.Div(
        [
            html.H1("Log Page"),
            dbc.Card(
                [
                    dbc.CardHeader("Query Logs"),
                    dbc.CardBody(
                        [
                            html.Pre(
                                id="log-contents",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "border": "1px solid #333333",
                                },
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


@callback(
    Output("log-contents", "children"),
    Input("url", "pathname"),
)
def display_logs(pathname):
    """Display logs."""
    if dash_store.engine_instance is None:
        return "No logs found."
    else:
        e = dash_store.engine_instance
        log_file_path = e.log_file_path
        logger.debug(f"Log file path: {log_file_path}")
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                log_contents = f.read()
        else:
            log_contents = "No logs found."
        return log_contents
