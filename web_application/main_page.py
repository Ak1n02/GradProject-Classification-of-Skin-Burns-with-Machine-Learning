import streamlit as st

st.set_page_config(
    page_title="Main Page",
)

# Integrate CSS styles
with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def notification_content():
    st.markdown("""
    <div class="notification-box">
        <div class="warning-message">
            Bu uygulamayı kullanmadan önce onay vermeniz gerekiyor.
        </div>
        <div class="notification-title">
            Yanık Tespit Uygulamamızın Çalışma Koşulları
        </div>
        <div class="notification-content">
            Uygulamamızın doğru bir şekilde çalışabilmesi için aşağıdaki koşullara uymak gerekmektedir: <br><br>
            📌 Yüklenen resimlerin yanık içermesi gerekmektedir.<br>
            📌 Yüklenen resimlerin çok fazla gürültü içermemesi gerekmemektedir.<br>
            📌 Bunlarla birlikte yüklenen resimlerin net, çok parlak olmaması gerekmektedir.<br><br>
            Eğer yukarıda bahsettiğimiz kurallara uyulmazsa uygulamamız düzgün çalışmayabilir.
        </div>
    </div>
    """, unsafe_allow_html=True)

def modify_verified():
    st.session_state.verified = True

# First part --> inform the user about the application
if 'verified' not in st.session_state:
    st.session_state.verified = False

if not st.session_state.verified:

    notification_content()
    st.button("Onayla", key="verify", on_click=modify_verified)







