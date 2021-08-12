import streamlit as st
from multiapp import MultiApp
from apps import op45_46_sing_plat, op45_46_plat_comp, op90_plat_comp , op180_185_sing_plat, op180_185_plat_comp, all_lines_analysis, capability_analysis# import your app modules here

st.set_page_config(page_title='Plant Production Dashboard')

app = MultiApp()

# Add all your application here
app.add_app("Site 2 Overview", all_lines_analysis.app)
app.add_app("OP45-46 Platform Comparison", op45_46_plat_comp.app)
app.add_app("OP45-46 Single Platform", op45_46_sing_plat.app)
app.add_app("OP180-185 Platform Comparison", op180_185_plat_comp.app)
app.add_app("OP180-185 Single Platform", op180_185_sing_plat.app)
app.add_app("OP90 All", op90_plat_comp.app)
app.add_app("Capability Study", capability_analysis.app)


# The main app
app.run()
