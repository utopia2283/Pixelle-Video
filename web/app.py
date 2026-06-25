# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pixelle-Video Web UI - Main Entry Point

This is the entry point for the Streamlit multi-page application.
Uses st.navigation to define pages and set the default page to Home.
"""

import sys
from pathlib import Path

# Add project root to sys.path for module imports
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

# Setup page config (must be first Streamlit command)
st.set_page_config(
    page_title="Pixelle-Video - AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _require_access_code():
    """Gate the whole app behind a shared team access code (env PIXELLE_ACCESS_CODE).

    No-op when the env var is unset, so local/dev use is unaffected. When set,
    the user must enter the matching code once per browser session before any
    page renders.
    """
    import os
    code = os.getenv("PIXELLE_ACCESS_CODE", "").strip()
    if not code or st.session_state.get("_pv_authed"):
        return
    st.markdown("## 🔒 Pixelle-Video")
    st.caption("請輸入團隊 Access Code")
    entered = st.text_input(
        "Access Code", type="password", key="_pv_code",
        label_visibility="collapsed", placeholder="Access Code",
    )
    if entered:
        if entered == code:
            st.session_state["_pv_authed"] = True
            st.rerun()
        else:
            st.error("Access code 唔啱，請再試。")
    st.stop()


def main():
    """Main entry point with navigation"""
    _require_access_code()
    # Define pages using st.Page
    home_page = st.Page(
        "pages/1_🎬_Home.py",
        title="Home",
        icon="🎬",
        default=True
    )
    
    history_page = st.Page(
        "pages/2_📚_History.py",
        title="History",
        icon="📚"
    )
    
    # Set up navigation and run
    pg = st.navigation([home_page, history_page])
    pg.run()


if __name__ == "__main__":
    main()
