import streamlit as st

st.set_page_config(
    page_title="Ayudantía Streamilt",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.markdown('<h1 class="main-header">Ayundatía: Streamlit</h1>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.title("Barra lateral")

    st.markdown("link ejemplos: https://streamlit.io/gallery")



# Ejecutar la aplicación
if __name__ == "__main__":
    main()