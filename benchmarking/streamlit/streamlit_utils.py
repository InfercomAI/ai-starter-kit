import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.graph_objs import Figure

from benchmarking.utils import INFERCOM_API_BASE
from utils.visual.env_utils import are_credentials_set, env_input_fields, initialize_env_variables, save_credentials

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

# === Infercom Brand Colors (from Infercom_Color_System.pdf) ===
BRAND_GREEN = '#1FA85F'
BRAND_GREEN_LIGHT = '#40B577'
BRAND_GREEN_DARK = '#18864C'
BRAND_GREEN_EXTRA_LIGHT = '#78CA9F'

BRAND_BLUE = '#0B7FDE'
BRAND_BLUE_DARK = '#0865B1'

BRAND_ORANGE = '#E67E22'
BRAND_ORANGE_DARK = '#B8641B'

BRAND_CHARCOAL = '#2D2D2D'
BRAND_CHARCOAL_LIGHT = '#4C4C4C'
BRAND_CHARCOAL_DARK = '#242424'

BRAND_TEXT = '#FAFAFA'
BRAND_TEXT_SECONDARY = '#E0E0E0'

# Chart colors: Server=Blue (technology), Client=Orange (performance)
CHART_COLOR_SERVER = BRAND_BLUE
CHART_COLOR_CLIENT = BRAND_ORANGE

# Data visualization color sequence (official brand order)
BRAND_DATAVIZ_COLORS = [BRAND_GREEN, BRAND_BLUE, BRAND_ORANGE, BRAND_GREEN_LIGHT, '#2F92E2']

# Brand assets CDN
BRAND_CDN = 'https://infercomai.github.io/brand-assets'
BRAND_FAVICON = f'{BRAND_CDN}/favicons/favicon-96x96.png'
BRAND_LOGO_WHITE = f'{BRAND_CDN}/logos/infercom-logo-white-400px.png'

LLM_API_OPTIONS = {'sncloud': 'Infercom Inference Service'}
MULTIMODAL_IMAGE_SIZE_OPTIONS = {'na': 'N/A', 'small': 'Small', 'medium': 'Medium', 'large': 'Large'}
QPS_DISTRIBUTION_OPTIONS = {'constant': 'Constant', 'uniform': 'Uniform', 'exponential': 'Exponential'}

# Default model to show in the dropdown
DEFAULT_MODEL = 'gpt-oss-120b'

# Region display labels for the model selector
_REGION_FLAG = {
    'EU': '\U0001F1EA\U0001F1FA',  # 🇪🇺
}
_GLOBAL_FLAG = '\U0001F310'  # 🌐


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_available_models() -> List[str]:
    """Fetch available models from the Infercom API, grouped by region.

    EU sovereign models are listed first, followed by Global Catalog models.

    Returns:
        List of model IDs available for inference.
    """
    try:
        response = requests.get(
            f'{INFERCOM_API_BASE}/models?verbose=true',
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        eu_models = []
        global_models = []
        for model in data.get('data', []):
            model_id = model['id']
            region = (model.get('sn_metadata') or {}).get('region', '')
            if region == 'EU':
                eu_models.append(model_id)
            else:
                global_models.append(model_id)

        eu_models.sort()
        global_models.sort()

        # EU models first, then global — default model at the top of its group
        if DEFAULT_MODEL in eu_models:
            eu_models.remove(DEFAULT_MODEL)
            eu_models.insert(0, DEFAULT_MODEL)
        elif DEFAULT_MODEL in global_models:
            global_models.remove(DEFAULT_MODEL)
            global_models.insert(0, DEFAULT_MODEL)

        models = eu_models + global_models

        # Store region metadata in session state for format_func
        region_map = {}
        for model in data.get('data', []):
            region = (model.get('sn_metadata') or {}).get('region', '')
            region_map[model['id']] = region
        st.session_state['_model_regions'] = region_map

        return models if models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


MODEL_SELECTOR_HELP = (
    '\U0001F1EA\U0001F1FA EU Sovereign — hosted in Infercom\'s EU datacenter. '
    '\U0001F310 Global Catalog — routed via global infrastructure. '
    '[Learn more](https://docs.infercom.ai/en/models/infercomcloud-models#identifying-model-regions)'
)


def format_model_name(model_id: str) -> str:
    """Format a model ID for display in the selector with region flag.

    Use as format_func in st.selectbox to show flags without altering the value.
    """
    region_map = st.session_state.get('_model_regions', {})
    region = region_map.get(model_id, '')
    if region == 'EU':
        return f'{_REGION_FLAG["EU"]} {model_id}'
    elif region:
        return f'{_GLOBAL_FLAG} {model_id} ({region})'
    return model_id


APP_PAGES = {
    'synthetic_eval': {
        'file_path': 'pages/synthetic_performance_eval_st.py',
        'page_label': 'Synthetic Performance Evaluation',
        'page_icon': ':material/analytics:',
    },
    'real_workload_eval': {
        'file_path': 'pages/real_workload_eval_st.py',
        'page_label': 'Real Workload Evaluation',
        'page_icon': ':material/speed:',
    },
    'custom_eval': {
        'file_path': 'pages/custom_performance_eval_st.py',
        'page_label': 'Custom Performance Evaluation',
        'page_icon': ':material/instant_mix:',
    },
    'chat_eval': {
        'file_path': 'pages/chat_performance_st.py',
        'page_label': 'Performance on Chat',
        'page_icon': ':material/chat:',
    },
}


def render_logo() -> None:
    """Render the Infercom logo in the sidebar, loaded from the brand assets CDN."""
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <a href="https://www.infercom.ai" target="_blank" rel="noopener noreferrer">
                <img src="{BRAND_LOGO_WHITE}"
                     alt="Infercom"
                     style="width: 60%; display: block; margin: 0 auto; max-width: 100%;">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_font() -> None:
    """Load Roboto font from Google Fonts and apply brand styling globally."""
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">

        <style>
            /* Infercom Brand CSS Custom Properties */
            :root {{
                --infercom-green: {BRAND_GREEN};
                --infercom-green-light: {BRAND_GREEN_LIGHT};
                --infercom-green-dark: {BRAND_GREEN_DARK};
                --infercom-blue: {BRAND_BLUE};
                --infercom-orange: {BRAND_ORANGE};
                --infercom-charcoal: {BRAND_CHARCOAL};
                --infercom-charcoal-dark: {BRAND_CHARCOAL_DARK};
                --infercom-text: {BRAND_TEXT};
                --infercom-text-secondary: {BRAND_TEXT_SECONDARY};
            }}

            /* Apply Roboto font globally */
            html, body, [class^="css"] :not(.material-icons) {{
                font-family: 'Roboto', sans-serif !important;
            }}

            /* ── Sidebar styling (Green Dark brand color) ── */
            section[data-testid="stSidebar"] {{
                background-color: {BRAND_GREEN_DARK} !important;
                border-right: 1px solid {BRAND_GREEN}44 !important;
            }}

            /* Sidebar section headers — white left accent bar on green bg */
            section[data-testid="stSidebar"] .stHeading h1,
            section[data-testid="stSidebar"] .stHeading h2,
            section[data-testid="stSidebar"] .stHeading h3,
            section[data-testid="stSidebar"] .stMarkdown h1,
            section[data-testid="stSidebar"] .stMarkdown h2,
            section[data-testid="stSidebar"] .stMarkdown h3 {{
                border-left: 3px solid #FFFFFF !important;
                padding-left: 0.6rem !important;
                margin-top: 1.2rem !important;
                margin-bottom: 0.4rem !important;
                font-weight: 500 !important;
                letter-spacing: -0.01em !important;
                color: #FFFFFF !important;
            }}

            /* Nav separator — white translucent line on green bg */
            [data-testid="stSidebarNavSeparator"] {{
                border-bottom: none !important;
                background: linear-gradient(90deg, rgba(255,255,255,0.25), transparent) !important;
                height: 1px !important;
                margin: 0.5rem 0 !important;
            }}

            /* Sidebar horizontal rules */
            section[data-testid="stSidebar"] hr {{
                border: none !important;
                height: 1px !important;
                background: linear-gradient(90deg, rgba(255,255,255,0.25), transparent) !important;
                margin: 0.8rem 0 !important;
            }}

            /* Sidebar text — ensure white on green */
            section[data-testid="stSidebar"] {{
                color: #FFFFFF !important;
            }}
            section[data-testid="stSidebar"] .stMarkdown {{
                color: #FFFFFF !important;
            }}
            section[data-testid="stSidebar"] .stMarkdown p,
            section[data-testid="stSidebar"] .stMarkdown span {{
                color: #FFFFFF !important;
            }}
            section[data-testid="stSidebar"] .stMarkdown a {{
                color: {BRAND_GREEN_EXTRA_LIGHT} !important;
            }}

            /* ── Input fields ── */
            /* Sidebar inputs: charcoal dark on green sidebar */
            section[data-testid="stSidebar"] .stTextInput input,
            section[data-testid="stSidebar"] .stNumberInput input {{
                background-color: {BRAND_CHARCOAL_DARK} !important;
                border: 1px solid rgba(255,255,255,0.2) !important;
                border-radius: 6px !important;
                color: {BRAND_TEXT} !important;
                transition: border-color 0.2s ease !important;
            }}
            /* Main area inputs */
            .stTextInput input,
            .stNumberInput input {{
                background-color: {BRAND_CHARCOAL_DARK} !important;
                border: 1px solid {BRAND_CHARCOAL_LIGHT} !important;
                border-radius: 6px !important;
                color: {BRAND_TEXT} !important;
                transition: border-color 0.2s ease !important;
            }}

            /* Input focus state */
            section[data-testid="stSidebar"] .stTextInput input:focus,
            section[data-testid="stSidebar"] .stNumberInput input:focus {{
                border-color: {BRAND_GREEN_EXTRA_LIGHT} !important;
                box-shadow: 0 0 0 1px {BRAND_GREEN_EXTRA_LIGHT}40 !important;
            }}
            .stTextInput input:focus,
            .stNumberInput input:focus {{
                border-color: {BRAND_GREEN} !important;
                box-shadow: 0 0 0 1px {BRAND_GREEN}40 !important;
            }}

            /* Selectbox container — dark on green sidebar */
            section[data-testid="stSidebar"] [data-baseweb="select"] {{
                background-color: {BRAND_CHARCOAL_DARK} !important;
                border-radius: 6px !important;
            }}
            section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
                background-color: {BRAND_CHARCOAL_DARK} !important;
                border-color: rgba(255,255,255,0.2) !important;
                border-radius: 6px !important;
            }}
            section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {{
                border-color: {BRAND_GREEN_EXTRA_LIGHT} !important;
            }}

            /* Selectbox dropdown menu */
            [data-baseweb="popover"] {{
                background-color: {BRAND_CHARCOAL_DARK} !important;
                border: 1px solid {BRAND_CHARCOAL_LIGHT} !important;
                border-radius: 6px !important;
            }}
            [data-baseweb="popover"] li {{
                background-color: {BRAND_CHARCOAL_DARK} !important;
            }}
            [data-baseweb="popover"] li:hover {{
                background-color: {BRAND_GREEN_DARK} !important;
            }}
            [role="option"][aria-selected="true"] {{
                background-color: {BRAND_GREEN_DARK} !important;
            }}

            /* Number input +/- buttons on green sidebar */
            section[data-testid="stSidebar"] .stNumberInput button {{
                border-color: rgba(255,255,255,0.2) !important;
                color: #FFFFFF !important;
                background-color: {BRAND_CHARCOAL_DARK} !important;
            }}
            section[data-testid="stSidebar"] .stNumberInput button:hover {{
                border-color: {BRAND_GREEN_EXTRA_LIGHT} !important;
                color: {BRAND_GREEN_EXTRA_LIGHT} !important;
            }}
            .stNumberInput button {{
                border-color: {BRAND_CHARCOAL_LIGHT} !important;
                color: {BRAND_TEXT_SECONDARY} !important;
            }}
            .stNumberInput button:hover {{
                border-color: {BRAND_GREEN} !important;
                color: {BRAND_GREEN} !important;
            }}

            /* ── Labels ── */
            /* Sidebar labels: white on green */
            section[data-testid="stSidebar"] .stSelectbox label,
            section[data-testid="stSidebar"] .stTextInput label,
            section[data-testid="stSidebar"] .stNumberInput label,
            section[data-testid="stSidebar"] .stSlider label {{
                color: rgba(255,255,255,0.85) !important;
                font-size: 0.85rem !important;
                font-weight: 400 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.04em !important;
            }}
            /* Main area labels */
            .stSelectbox label,
            .stTextInput label,
            .stNumberInput label,
            .stSlider label {{
                color: {BRAND_TEXT_SECONDARY} !important;
                font-size: 0.85rem !important;
                font-weight: 400 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.04em !important;
            }}

            /* ── Main content area ── */
            .stMainBlockContainer {{
                background-color: {BRAND_CHARCOAL} !important;
            }}

            /* ── Expander styling ── */
            .streamlit-expanderHeader {{
                background-color: {BRAND_CHARCOAL_DARK} !important;
                border-radius: 6px !important;
                border: 1px solid {BRAND_CHARCOAL_LIGHT} !important;
            }}

            /* ── Metric cards ── */
            [data-testid="stMetric"] {{
                background-color: {BRAND_CHARCOAL_DARK} !important;
                border: 1px solid {BRAND_CHARCOAL_LIGHT} !important;
                border-radius: 8px !important;
                padding: 0.8rem 1rem !important;
            }}
            [data-testid="stMetric"] [data-testid="stMetricValue"] {{
                color: {BRAND_GREEN_LIGHT} !important;
                font-weight: 700 !important;
            }}

            /* ── Tabs ── */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0 !important;
                border-bottom: 1px solid {BRAND_CHARCOAL_LIGHT} !important;
            }}
            .stTabs [data-baseweb="tab"] {{
                border-bottom: 2px solid transparent !important;
                color: {BRAND_TEXT_SECONDARY} !important;
                padding: 0.5rem 1rem !important;
            }}
            .stTabs [aria-selected="true"] {{
                border-bottom-color: {BRAND_GREEN} !important;
                color: {BRAND_TEXT} !important;
            }}

            /* ── Data table styling ── */
            .stDataFrame {{
                border: 1px solid {BRAND_CHARCOAL_LIGHT} !important;
                border-radius: 6px !important;
            }}

            /* ── Toast / alert messages ── */
            .stSuccess {{
                background-color: {BRAND_GREEN}15 !important;
                border-left: 3px solid {BRAND_GREEN} !important;
                color: {BRAND_TEXT} !important;
            }}

            /* ── Scrollbar styling ── */
            ::-webkit-scrollbar {{
                width: 6px;
                height: 6px;
            }}
            ::-webkit-scrollbar-track {{
                background: {BRAND_CHARCOAL_DARK};
            }}
            ::-webkit-scrollbar-thumb {{
                background: {BRAND_CHARCOAL_LIGHT};
                border-radius: 3px;
            }}
            ::-webkit-scrollbar-thumb:hover {{
                background: {BRAND_GREEN}88;
            }}

            /* ── Primary button styling ── */
            /* On green sidebar: white button */
            section[data-testid="stSidebar"] .stButton > button[kind="primary"],
            section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{
                background-color: #FFFFFF !important;
                border-color: #FFFFFF !important;
                color: {BRAND_GREEN_DARK} !important;
                font-family: 'Roboto', sans-serif !important;
                font-weight: 600 !important;
                border-radius: 6px !important;
                transition: all 0.2s ease !important;
            }}
            section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
            section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {{
                background-color: {BRAND_GREEN_EXTRA_LIGHT} !important;
                border-color: {BRAND_GREEN_EXTRA_LIGHT} !important;
                color: {BRAND_GREEN_DARK} !important;
            }}
            /* Main area: green button */
            .stButton > button[kind="primary"],
            button[data-testid="stBaseButton-primary"] {{
                background-color: {BRAND_GREEN} !important;
                border-color: {BRAND_GREEN} !important;
                color: #FFFFFF !important;
                font-family: 'Roboto', sans-serif !important;
                font-weight: 500 !important;
                border-radius: 6px !important;
                transition: all 0.2s ease !important;
            }}
            .stButton > button[kind="primary"]:hover,
            button[data-testid="stBaseButton-primary"]:hover {{
                background-color: {BRAND_GREEN_DARK} !important;
                border-color: {BRAND_GREEN_DARK} !important;
                box-shadow: 0 2px 8px {BRAND_GREEN}30 !important;
            }}

            /* Secondary button styling — white outline on green sidebar */
            section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
            section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
                border-color: #FFFFFF !important;
                color: #FFFFFF !important;
                font-family: 'Roboto', sans-serif !important;
                font-weight: 500 !important;
                border-radius: 6px !important;
                background-color: transparent !important;
                transition: all 0.2s ease !important;
            }}
            section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
            section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {{
                border-color: {BRAND_GREEN_EXTRA_LIGHT} !important;
                color: {BRAND_GREEN_EXTRA_LIGHT} !important;
            }}
            /* Secondary button in main area — green outline */
            .stButton > button[kind="secondary"],
            button[data-testid="stBaseButton-secondary"] {{
                border-color: {BRAND_GREEN} !important;
                color: {BRAND_GREEN} !important;
                font-family: 'Roboto', sans-serif !important;
                font-weight: 500 !important;
                border-radius: 6px !important;
                background-color: transparent !important;
                transition: all 0.2s ease !important;
            }}
            .stButton > button[kind="secondary"]:hover,
            button[data-testid="stBaseButton-secondary"]:hover {{
                border-color: {BRAND_GREEN_LIGHT} !important;
                color: {BRAND_GREEN_LIGHT} !important;
                box-shadow: 0 0 0 1px {BRAND_GREEN}30 !important;
            }}

            /* Download button styling (blue) */
            .stDownloadButton > button {{
                background-color: {BRAND_BLUE} !important;
                border-color: {BRAND_BLUE} !important;
                color: #FFFFFF !important;
                font-family: 'Roboto', sans-serif !important;
                border-radius: 6px !important;
                transition: all 0.2s ease !important;
            }}
            .stDownloadButton > button:hover {{
                background-color: {BRAND_BLUE_DARK} !important;
                border-color: {BRAND_BLUE_DARK} !important;
                box-shadow: 0 2px 8px {BRAND_BLUE}30 !important;
            }}

            /* Progress bar brand color */
            .stProgress > div > div > div {{
                background-color: {BRAND_GREEN} !important;
            }}

            /* ── Navigation styling on green sidebar ── */
            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
                border-radius: 6px !important;
                transition: background-color 0.15s ease !important;
                color: #FFFFFF !important;
            }}
            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
                background-color: rgba(255,255,255,0.1) !important;
            }}
            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] {{
                background-color: rgba(255,255,255,0.15) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_title_icon(title: str, icon: Optional[str] = None) -> None:
    """Render a branded page title with optional icon."""
    if icon is not None:
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            st.image(icon)
    st.markdown(
        f"""
        <style>
            .kit-title {{
                text-align: center;
                color: {BRAND_GREEN_LIGHT} !important;
                font-size: 2.6em;
                font-weight: 700;
                font-family: 'Roboto', sans-serif !important;
                margin-bottom: 0.2em;
                letter-spacing: -0.01em;
            }}
            .kit-subtitle {{
                text-align: center;
                color: {BRAND_TEXT_SECONDARY} !important;
                font-size: 0.95em;
                font-weight: 300;
                font-family: 'Roboto', sans-serif !important;
                margin-bottom: 1.5em;
            }}
        </style>
        <div class="kit-title">{title}</div>
        <div class="kit-subtitle">Powered by Infercom &mdash; EU Sovereign AI Inference</div>
        """,
        unsafe_allow_html=True,
    )


def setup_credentials() -> None:
    """Sets up the credentials for the application."""

    st.title('Setup')

    # Callout to get Infercom API Key
    st.markdown('Get your Infercom API key [here](https://cloud.infercom.ai/apis)')

    # Set the llm_api to sncloud (only option for now)
    st.session_state.llm_api = 'sncloud'

    additional_env_vars: Dict[str, Any] = {}
    additional_env_vars = {'INFERCOM_API_BASE': INFERCOM_API_BASE}

    initialize_env_variables(st.session_state.prod_mode, additional_env_vars)

    if not are_credentials_set():
        api_key, additional_vars = env_input_fields(additional_env_vars)
        if st.button('Save Credentials', key='save_credentials_sidebar'):
            message = save_credentials(api_key, additional_vars, st.session_state.prod_mode)
            st.session_state.mp_events.api_key_saved()
            st.success(message)
            st.rerun()
    else:
        st.success('Credentials are set')
        if st.button('Clear Credentials', key='clear_credentials'):
            if st.session_state.llm_api == 'sncloud':
                save_credentials('', None, st.session_state.prod_mode)
            else:
                save_credentials('', {var: '' for var in additional_env_vars}, st.session_state.prod_mode)
            st.rerun()


def save_uploaded_file(internal_save_path: str) -> str:
    uploaded_file = st.session_state.uploaded_file
    temp_file_path = '.'
    if st.session_state.uploaded_file is not None:
        # Save the uploaded file to a temporary location
        save_dir = os.path.join(os.getcwd(), internal_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        temp_file_path = os.path.join(save_dir, uploaded_file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
    return temp_file_path


def find_pages_to_show() -> List[Any]:
    pages = st.session_state.pages_to_show
    pages_to_show = []

    for page_k, _ in APP_PAGES.items():
        if page_k in pages:
            pages_to_show.append(
                st.Page(
                    APP_PAGES[page_k]['file_path'],
                    title=APP_PAGES[page_k]['page_label'],
                    icon=APP_PAGES[page_k]['page_icon'],
                )
            )
    return pages_to_show


def update_progress_bar(step: int, total_steps: int) -> None:
    """Update the progress bar."""
    st.session_state.progress_bar.progress(value=step / total_steps, text=f'Running requests: {step}/{total_steps}')


def set_api_variables() -> Dict[str, Any]:
    if st.session_state.prod_mode:
        # Infercom Inference Service
        if st.session_state.llm_api == 'sncloud':
            # Use input field value directly, fall back to saved value
            api_key = st.session_state.get('api_key_input', '') or st.session_state.get('INFERCOM_API_KEY', '')
            api_variables = {
                'INFERCOM_API_BASE': st.session_state.INFERCOM_API_BASE,
                'INFERCOM_API_KEY': api_key,
            }
        else:
            raise Exception('Only sncloud supported.')
    else:
        api_variables = {}

    return api_variables


def _get_infercom_plotly_template() -> go.layout.Template:
    """Return a custom Plotly template with Infercom brand colors."""
    return go.layout.Template(
        layout={
            'paper_bgcolor': BRAND_CHARCOAL,
            'plot_bgcolor': BRAND_CHARCOAL_DARK,
            'font': {
                'family': 'Roboto, sans-serif',
                'color': BRAND_TEXT,
            },
            'title': {
                'font': {
                    'family': 'Roboto, sans-serif',
                    'color': BRAND_TEXT,
                    'size': 16,
                },
            },
            'xaxis': {
                'gridcolor': BRAND_CHARCOAL_LIGHT,
                'zerolinecolor': BRAND_CHARCOAL_LIGHT,
            },
            'yaxis': {
                'gridcolor': BRAND_CHARCOAL_LIGHT,
                'zerolinecolor': BRAND_CHARCOAL_LIGHT,
            },
            'colorway': BRAND_DATAVIZ_COLORS,
            'legend': {
                'font': {
                    'color': BRAND_TEXT_SECONDARY,
                },
            },
        }
    )


INFERCOM_PLOTLY_TEMPLATE = _get_infercom_plotly_template()


def plot_dataframe_summary(df_req_info: pd.DataFrame) -> Figure:
    """
    Plots a throughput summary across all batch sizes

    Args:
        df_req_info (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        fig (go.Figure): The plotly figure container
    """
    df_req_summary = (
        df_req_info.groupby('batch_size_used')[
            [
                'server_output_token_per_s_per_request',
                'client_output_token_per_s_per_request',
            ]
        ]
        .mean()
        .reset_index()
    ).rename(
        columns={
            'server_output_token_per_s_per_request': 'server_output_token_per_s_mean',
            'client_output_token_per_s_per_request': 'client_output_token_per_s_mean',
        }
    )
    df_req_summary['server_throughput_token_per_s'] = (
        df_req_summary['server_output_token_per_s_mean'] * df_req_summary['batch_size_used']
    )
    df_req_summary['client_throughput_token_per_s'] = (
        df_req_summary['client_output_token_per_s_mean'] * df_req_summary['batch_size_used']
    )
    df_req_summary.rename(
        columns={
            'batch_size_used': 'Batch size',
            'server_throughput_token_per_s': 'Server',
            'client_throughput_token_per_s': 'Client',
        },
        inplace=True,
    )
    df_melted = pd.melt(
        df_req_summary,
        id_vars='Batch size',
        value_vars=['Server', 'Client'],
        var_name='Side type',
        value_name='Total output throughput (tokens per second)',
    )

    df_melted['Total output throughput (tokens per second)'] = df_melted[
        'Total output throughput (tokens per second)'
    ].round(2)

    df_melted['Batch size'] = [str(x) for x in df_melted['Batch size']]
    fig = px.bar(
        df_melted,
        x='Batch size',
        y='Total output throughput (tokens per second)',
        color='Side type',
        barmode='group',
        color_discrete_sequence=[CHART_COLOR_SERVER, CHART_COLOR_CLIENT],
        text='Total output throughput (tokens per second)',
    )

    fig.update_traces(textposition='outside')  # Set text position outside bars

    fig.update_layout(
        title_text='Total output throughput per batch size',
        template=INFERCOM_PLOTLY_TEMPLATE,
    )
    return fig


def plot_client_vs_server_barplots(
    df_user: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    legend_labels: List[str],
    title: str,
    ylabel: str,
    xlabel: str,
    batching_exposed: bool,
) -> Figure:
    """
    Plots bar plots for client vs server metrics from a DataFrame.

    Args:
        df_user (pd.DataFrame): The DataFrame containing the data to plot.
        x_col (str): The column name to be used as the x-axis.
        y_cols (List[str]): A list of column names to be used as the y-axis.
        legend_labels (List[str]): Human-readable labels for each grouping in y_cols.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        xlabel (str): The label for the x-axis.
        batching_exposed (bool): boolean identifying if batching was exposed.

    Returns:
        fig (go.Figure): The plotly figure container
    """
    value_vars = y_cols
    title_text = title
    yaxis_title = ylabel
    xaxis_title = xlabel if batching_exposed else ''

    df_melted = df_user.melt(
        id_vars=[x_col],
        value_vars=value_vars,
        var_name='Metric',
        value_name='Value',
    )
    xgroups = [str(x) for x in sorted(pd.unique(df_melted[x_col]))]
    df_melted[x_col] = [str(x) for x in df_melted[x_col]]

    valsl = {}
    valsr = {}
    for i in xgroups:
        maskl = (df_melted['Metric'] == value_vars[0]) & (df_melted[x_col] == i)
        valsl[i] = np.percentile(df_melted['Value'][maskl], [5, 50, 95])
        maskr = (df_melted['Metric'] == value_vars[1]) & (df_melted[x_col] == i)
        valsr[i] = np.percentile(df_melted['Value'][maskr], [5, 50, 95])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=xgroups,
            y=[0 for _ in xgroups],
            base=[round(valsl[i][1], 2) for i in xgroups],
            customdata=[legend_labels[0] for _ in xgroups],
            marker={'color': CHART_COLOR_SERVER, 'line': {'color': CHART_COLOR_SERVER, 'width': 2}},
            offsetgroup=0,
            legendgroup=legend_labels[0],
            name=legend_labels[0],
            showlegend=False,
            hovertemplate='<extra></extra><b>%{customdata}</b> median: %{base:.2f}',
            text=[round(valsl[i][1], 2) for i in xgroups],
            textposition='outside',
        )
    )
    fig.add_trace(
        go.Bar(
            x=xgroups,
            y=[valsl[i][2] - valsl[i][0] for i in xgroups],
            base=[valsl[i][0] for i in xgroups],
            customdata=[valsl[i][2] for i in xgroups],
            marker={'color': CHART_COLOR_SERVER},
            opacity=0.5,
            offsetgroup=0,
            legendgroup=legend_labels[0],
            name=legend_labels[0],
            hovertemplate='<extra></extra>5–95 pctile range: %{base:.2f}–%{customdata:.2f}',
        )
    )
    fig.add_trace(
        go.Bar(
            x=xgroups,
            y=[0 for _ in xgroups],
            base=[round(valsr[i][1], 2) for i in xgroups],
            customdata=[legend_labels[1] for _ in xgroups],
            marker={'color': CHART_COLOR_CLIENT, 'line': {'color': CHART_COLOR_CLIENT, 'width': 2}},
            offsetgroup=1,
            legendgroup=legend_labels[1],
            name=legend_labels[1],
            showlegend=False,
            hovertemplate='<extra></extra><b>%{customdata}</b> median: %{base:.2f}',
            text=[round(valsr[i][1], 2) for i in xgroups],
            textposition='outside',
        )
    )
    fig.add_trace(
        go.Bar(
            x=xgroups,
            y=[valsr[i][2] - valsr[i][0] for i in xgroups],
            base=[valsr[i][0] for i in xgroups],
            customdata=[valsr[i][2] for i in xgroups],
            marker={'color': CHART_COLOR_CLIENT},
            opacity=0.5,
            offsetgroup=1,
            legendgroup=legend_labels[1],
            name=legend_labels[1],
            hovertemplate='<extra></extra>5–95 pctile range: %{base:.2f}–%{customdata:.2f}',
        )
    )

    fig.update_layout(
        title_text=title_text,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        barmode='group',
        template=INFERCOM_PLOTLY_TEMPLATE,
        hovermode='x unified',
    )

    fig.update_xaxes(hoverformat='foo', showticklabels=batching_exposed)

    return fig


def plot_requests_gantt_chart(df_user: pd.DataFrame) -> Figure:
    """
    Plots a Gantt chart of response timings across all requests

    Args:
        df_user (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        fig (go.Figure): The plotly figure container
    """
    requests = df_user.index + 1
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=requests,
            x=1000 * df_user['client_ttft_s'],
            base=[str(x) for x in df_user['start_time']],
            name='TTFT',
            orientation='h',
            marker_color=CHART_COLOR_CLIENT,
        )
    )
    fig.add_trace(
        go.Bar(
            y=requests,
            x=1000 * df_user['client_end_to_end_latency_s'],
            base=[str(x) for x in df_user['start_time']],
            name='End-to-end latency',
            orientation='h',
            marker_color=CHART_COLOR_SERVER,
        )
    )
    for i in range(0, len(df_user.index), 2):
        fig.add_hrect(y0=i + 0.5, y1=i + 1.5, line_width=0, fillcolor='grey', opacity=0.1)
    fig.update_xaxes(
        type='date',
        tickformat='%H:%M:%S',
        hoverformat='%H:%M:%S.%2f',
    )
    fig.update_layout(
        title_text='LLM requests across time',
        xaxis_title='Time stamp',
        yaxis_title='Request index',
        template=INFERCOM_PLOTLY_TEMPLATE,
    )
    return fig
