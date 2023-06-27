from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import streamlit  as st
from streamlit_chat import message
def logic(image,text):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
    return  (f"Predicted answer:, {model.config.id2label[idx]}")


def image_extractor(url):
    url = url
    image = Image.open(requests.get(url, stream=True,timeout=1000).raw)
    return image
    
def init():
    st.set_page_config(page_title=" Image processor ",
                       page_icon="📷")
    st.header("Ask any thing about the image")
    message("Hello ! Please Provide me the URL of the image")
    with st.sidebar:
        user_url=st.text_input("Enter the URl of the Image",key="image_url")
        user_query=st.text_input("enter the query",key="query")
    if user_url and user_query not in ["",None]:
        image=image_extractor(user_url)
        print("image received")
        new_image = image.resize((600, 400))
        st.image(new_image)
        # message(st.image(new_image),is_user=True)
        response=logic(image,user_query)
        message(response)





if __name__=="__main__":
    init()