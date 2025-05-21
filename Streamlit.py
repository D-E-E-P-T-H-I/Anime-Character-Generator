import streamlit as st #web framework
import torch #aiml operations
import torchvision.utils as vutils #image grids
import matplotlib.pyplot as plt #display images
from PlainGenerator import Generator #ai model
import base64 #image encoding

st.set_page_config(
    layout="wide",  #use full window width
    page_title="Anime Face Generator", #browser tab title
)

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image= get_base64_image("animeimage.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}"); 
        background-size: cover;
        background-position: center; 
        background-attachment: fixed;
    }}
    
    .header-overlay {{
        background: rgba(0, 0, 0, 0.7); /*Semi transparent black*/
        padding: 0.5rem; 
        border-radius: 20px 20px 20px 20px; /*Rounded corners*/
        margin-bottom: 3rem;
        backdrop-filter: blur(5px); /*Frosted glass effect*/
    }}
    
    .main-title {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(#7a6699, #ff4da6, #7a6699);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .subtitle {{
        color: white;
        text-align: center;
        font-size: 2rem;
        margin-bottom: 1.5rem;
        opacity: 0.9;
    }}
    
    .social-links {{
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 0.5rem;
    }}
    
    .social-link {{
        color: white !important;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.3s ease;
        font-weight: 500;
    }}
    
    .social-link:hover {{
        color: #ff66cc !important;
        transform: translateY(-2px);
        }}
    
    .stButton>button {{
        display: block;
        margin: 1.5rem auto 0;
        background: linear-gradient( #7a6699, #ff99cc, #7a6699);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 105, 180, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-3px);
    }}
    
    .stNumberInput>div>div>input {{
        background-color: rgba(255,255,255,0.9) !important;
        color: #333 !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
        border: 2px solid rgba(255,105,180,0.3) !important;
    }}
    
    .results-container {{
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(5px);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
@st.cache_resource
def load_generator():
    g = Generator()
    g.load_state_dict(torch.load("generator.pth", map_location=torch.device("cpu")))
    g.eval()
    return g

gen = load_generator()

st.markdown("""
<div class="header-overlay">
    <h1 class="main-title">Anime Character Generator</h1>
    <p class="subtitle">Create unique anime characters</p>
    <div class="social-links">
        <a href="https://github.com/D-E-E-P-T-H-I/Anime-Character-Generator" class="social-link" target="_blank">
            <i class="fab fa-github"></i> GitHub
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Control panel
with st.container():
    st.markdown('<div class="control-label">Number of characters to generate:</div>', unsafe_allow_html=True)
    n_imgs = st.number_input("", min_value=1, max_value=500, value=1, step=1, 
                           label_visibility="collapsed")
    
    if st.button("Generate Anime Faces"):
        with st.spinner('Generating your anime characters...'):
            noise = torch.randn(n_imgs,100, 1, 1)
            with torch.no_grad():
                fake = gen(noise).detach().cpu()
            
            # Display results
            grid = vutils.make_grid(fake, nrow=min(n_imgs, 5), normalize=True)
            fig, ax = plt.subplots(figsize=(min(n_imgs, 5) * 1.5, (n_imgs // 5 + 1) * 1.5))
            ax.imshow(grid.permute(1, 2, 0))
            ax.axis("off")
            fig.patch.set_facecolor("none")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


