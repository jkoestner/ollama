"""
Building plotly dashboard.

Builds plotly pages with call backs. There are 3 options the user has for running code.
1. Local running
2. Docker running
3. CLI running

To run locally:
1. cd into directory with app.py
2. run plotly dashboard - `python app.py`

The ascii text is generated using https://patorjk.com/software/taag/
with "standard font"
"""

import dash_bootstrap_components as dbc
import dash_extensions.enrich as dash
from dash_extensions.enrich import (
    DashProxy,
    Input,
    Output,
    callback,
    dcc,
    html,
)

from osllmh.dashboard.components import dash_store
from osllmh.utils import custom_logger

logger = custom_logger.setup_logging(__name__)


#      _    ____  ____
#     / \  |  _ \|  _ \
#    / _ \ | |_) | |_) |
#   / ___ \|  __/|  __/
#  /_/   \_\_|   |_|

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

app = DashProxy(
    __name__,
    use_pages=True,
    external_stylesheets=[
        # "https://codepen.io/chriddyp/pen/bWLwgP.css",
        dbc_css,
        dbc.themes.QUARTZ,
    ],
)
server = app.server
app.config.suppress_callback_exceptions = True

app.title = "osllmh Dashboard"
app._favicon = "llm_logo.ico"

# creating the navbar
page_links = [
    dbc.NavItem(dbc.NavLink(page["name"], href=page["relative_path"]))
    for page in dash.page_registry.values()
]

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src=app.get_asset_url("llm_logo.ico"), height="40px")
                    ),
                    dbc.Col(
                        [
                            dbc.NavbarBrand("osllmh", className="ms-3 fs-3"),
                            dbc.NavLink(
                                html.Img(
                                    src=app.get_asset_url("github-mark-white.png"),
                                    height="20px",
                                ),
                                href="https://github.com/jkoestner/osllmh",
                                target="_blank",
                                className="ms-2",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                        },
                    ),
                ],
                align="center",
                className="g-0",
            ),
            dbc.Row(
                dbc.Nav(
                    page_links,
                    className="ms-auto",
                    navbar=True,
                ),
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="primary",
    dark=True,
    sticky="top",
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="engine-store", storage_type="session", data=None),
        navbar,
        html.Div(id="status-div"),
        dash.page_container,
    ]
)


# update engine status
@callback(
    Output("status-div", "children"),
    [Input("url", "pathname"), Input("engine-store", "data")],
)
def update_status_div(pathname, settings):
    """Update loaded status on page refresh."""
    logger.debug("checking the engine status")
    engine_loaded = (
        "engine not loaded" if dash_store.engine_instance is None else "engine loaded"
    )
    badge_color = "success" if dash_store.engine_instance is not None else "secondary"

    status_div = html.H6(
        [dbc.Badge(engine_loaded, className="ms-1", color=badge_color)]
    )

    return status_div


if __name__ == "__main__":
    custom_logger.set_log_level("DEBUG", module_prefix="pages")
    custom_logger.set_log_level("DEBUG", module_prefix="__main__")
    app.run_server(debug=True)
