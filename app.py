import streamlit as st
import numpy as np
import components


st.set_page_config(
        page_title="GPCR-TargetSeeker"
    )
def app():
    st.markdown('''
                # :dart: GPCR TARGET SEEKER
                ''')
    st.write('*Use machine learning to predict the GPCR targets of your compounds*')

    tab1, tab2, tab3, tab4 = st.tabs(["GPCR Target Seeker", "Background", "Model Info", "Contact"])

    with tab1:
        components.ligand_qsar()
    with tab2:
        components.about()
    with tab3:
        components.model_info()
    with tab4:
        components.contact()

if __name__ == '__main__':
 
    app()