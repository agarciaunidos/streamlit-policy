import streamlit as st
# Asegúrate de que tus otros archivos no importen app.py de ninguna manera

# Define las funciones de navegación o importa las funciones de otros archivos aquí
def load_policy_app():
    from policy_app import main as policy_main
    policy_main()

def load_db_app():
    from app_dy import main as db_app
    db_app()

# Sidebar para la navegación
    

st.sidebar.image('https://unidosus.org/wp-content/themes/unidos/images/unidosus-logo-color-2x.png', use_column_width=True)
st.sidebar.title("Navigation")
pagina = st.sidebar.selectbox("Select a page:", ["Policy App", "Chat History"])

if pagina == "Policy App":
    load_policy_app()
elif pagina == "Chat History":
    load_db_app()
