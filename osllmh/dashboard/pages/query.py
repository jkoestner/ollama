"""Query dashboard."""

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
from dash_extensions.enrich import (
    Input,
    Output,
    State,
    callback,
    html,
)

from osllmh.dashboard.components import engine_store
from osllmh.utils import custom_logger

logger = custom_logger.setup_logging(__name__)

dash.register_page(__name__, path="/query", title="osllmh - Query", order=2)


#   _                            _
#  | |    __ _ _   _  ___  _   _| |_
#  | |   / _` | | | |/ _ \| | | | __|
#  | |__| (_| | |_| | (_) | |_| | |_
#  |_____\__,_|\__, |\___/ \__,_|\__|
#              |___/


def layout():
    """Query layout."""
    return html.Div(
        [
            html.H1("Query Page"),
            dbc.Card(
                [
                    dbc.CardHeader("Enter your query"),
                    dbc.CardBody(
                        [
                            dbc.Textarea(
                                id="query-input",
                                placeholder="Type your query here...",
                                style={"width": "100%", "height": 100},
                            ),
                            dbc.Button(
                                "Submit Query",
                                id="submit-query-button",
                                color="primary",
                                className="mt-2",
                            ),
                            html.Div(
                                id="query-response",
                                className="mt-4",
                                style={"whiteSpace": "pre-wrap"},
                            ),
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
    Output("query-response", "children"),
    Input("submit-query-button", "n_clicks"),
    State("query-input", "value"),
    prevent_initial_call=True,
)
def submit_query(n_clicks, query_text):
    """Submit query to engine."""
    if n_clicks is None or not query_text:
        return dash.no_update
    e = engine_store.engine_instance
    response = e.query(query_text)
    return response.response
