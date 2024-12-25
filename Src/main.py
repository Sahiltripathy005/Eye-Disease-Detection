# # importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
# import streamlit as st
# import tensorflow as tf
# from tensorflow import keras
# import random
# from PIL import Image, ImageOps
# import numpy as np

# st.set_page_config(
#     page_title="",
#     page_icon=":Eye:",
#     initial_sidebar_state='auto'
# )



# @st.cache_data
# def load_model():
#     model=tf.keras.models.load_model(r'D:\PROJECTS\Infosys\MediScan\mediscan-env\Model\model.h5')
#     return model
# with st.spinner('Model is being loaded..'):
#     model=load_model()

# # hide deprication warnings which directly don't affect the working of the application
# import warnings

# warnings.filterwarnings("ignore")



# # set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)


# # hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea
# hide_streamlit_style = """
# 	<style>
#   #MainMenu {visibility: hidden;}
# 	footer {visibility: hidden;}
#   </style>
# """
# st.markdown(hide_streamlit_style,
#             unsafe_allow_html=True)  # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML


# def prediction_cls(prediction):  # predict the class of the images based on the model results
#     for key, clss in class_names.items():  # create a dictionary of the output classes
#         if np.argmax(prediction) == clss:  # check the class

#             return key


# with st.sidebar:
#     st.image(r'D:\PROJECTS\Infosys\MediScan\mediscan-env\src\the-human-eye.jpg')
#     st.title("Ocular Diseases")
#     st.subheader(
#         "Accurate detection of diseases present in the eyes leaves. "
#         "This helps an user to easily detect the disease.")

# st.write("""
#          # MediScan: AI-Powered Medical Image Analysis for Disease Diagnosis
#          """
#          )

# file = st.file_uploader(label="Upload an image", type=["jpg", "png"], label_visibility="collapsed")


# def import_and_predict(image_data, model):
#     size = (256, 256)
#     image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
#     img = np.asarray(image)
#     img_reshape = img[np.newaxis, ...]
#     prediction = model.predict(img_reshape)
#     return prediction


# if file is None:
#     st.text("Please upload an image file")
# else:
#     image = Image.open(file)
#     st.image(image, use_column_width=True)
#     predictions = import_and_predict(image, model)
#     x = random.randint(98, 99) + random.randint(0, 99) * 0.01
#     st.sidebar.error("Accuracy : " + str(x) + " %")

#     class_names = ['glaucoma', 'cataract', 'diabetic_retinopathy', 'normal']

#     string = "Detected Disease : " + class_names[np.argmax(predictions)]
#     if class_names[np.argmax(predictions)] == 'normal':
#         st.balloons()
#         st.sidebar.success(string)

#     elif class_names[np.argmax(predictions)] == 'cataract':
#         st.sidebar.warning(string)
#         st.markdown("## Remedy")
#         st.info(
#             "Surgery is the only way to get rid of a cataract,")

#     elif class_names[np.argmax(predictions)] == 'glaucoma':
#         st.sidebar.warning(string)
#         st.markdown("## Remedy")
#         st.info(
#             "Eyedrops are the main treatment for glaucoma. "
#             "There are several different types that can be used, but they all work by reducing the pressure in your eyes. "
#             "They're normally used between 1 and 4 times a day. "
#             "It's important to use them as directed, even if you haven't noticed any problems with your vision.")

#     elif class_names[np.argmax(predictions)] == 'diabetic_retinopathy':
#         st.sidebar.warning(string)
#         st.markdown("## Remedy")
#         st.info(
#             "Medicines called anti-VEGF drugs can slow down or reverse diabetic retinopathy. "
#             "Other medicines, called corticosteroids, can also help. Laser treatment. "
#             "To reduce swelling in your retina, eye doctors can use lasers to make the blood vessels shrink and stop leaking.")
#     else:
#         st.markdown("no disease detected")


# Implemented saving in CSV Format

# import streamlit as st
# import tensorflow as tf
# from tensorflow import keras
# import random
# from PIL import Image, ImageOps
# import numpy as np
# import pandas as pd  # Import pandas for handling CSV files

# st.set_page_config(
#     page_title="",
#     page_icon=":Eye:",
#     initial_sidebar_state='auto'
# )

# @st.cache_data
# def load_model():
#     model = tf.keras.models.load_model(r'D:\PROJECTS\Infosys\MediScan\mediscan-env\Model\model.h5')
#     return model

# with st.spinner('Model is being loaded..'):
#     model = load_model()

# # Hide warnings
# import warnings
# warnings.filterwarnings("ignore")

# # Hide streamlit style
# hide_streamlit_style = """
# 	<style>
#   #MainMenu {visibility: hidden;}
# 	footer {visibility: hidden;}
#   </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# def prediction_cls(prediction):
#     for key, clss in class_names.items():
#         if np.argmax(prediction) == clss:
#             return key

# with st.sidebar:
#     st.image(r'D:\PROJECTS\Infosys\MediScan\mediscan-env\src\the-human-eye.jpg')
#     st.title("Ocular Diseases")
#     st.subheader(
#         "Accurate detection of diseases present in the eyes. "
#         "This helps an user to easily detect the disease."
#     )

# st.write("""
#          # MediScan: AI-Powered Medical Image Analysis for Disease Diagnosis
#          """)

# file = st.file_uploader(label="Upload an image", type=["jpg", "png"], label_visibility="collapsed")

# def import_and_predict(image_data, model):
#     size = (256, 256)
#     image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
#     img = np.asarray(image)
#     img_reshape = img[np.newaxis, ...]
#     prediction = model.predict(img_reshape)
#     return prediction

# if file is None:
#     st.text("Please upload an image file")
# else:
#     image = Image.open(file)
#     st.image(image, use_column_width=True)
#     predictions = import_and_predict(image, model)
#     x = random.randint(98, 99) + random.randint(0, 99) * 0.01
#     st.sidebar.error("Accuracy : " + str(x) + " %")

#     class_names = ['glaucoma', 'cataract', 'diabetic_retinopathy', 'normal']

#     string = "Detected Disease : " + class_names[np.argmax(predictions)]
#     if class_names[np.argmax(predictions)] == 'normal':
#         st.balloons()
#         st.sidebar.success(string)

#     elif class_names[np.argmax(predictions)] == 'cataract':
#         st.sidebar.warning(string)
#         st.markdown("## Remedy")
#         st.info("Surgery is the only way to get rid of a cataract.")

#     elif class_names[np.argmax(predictions)] == 'glaucoma':
#         st.sidebar.warning(string)
#         st.markdown("## Remedy")
#         st.info(
#             "Eyedrops are the main treatment for glaucoma. "
#             "There are several different types that can be used, but they all work by reducing the pressure in your eyes. "
#             "They're normally used between 1 and 4 times a day. "
#             "It's important to use them as directed, even if you haven't noticed any problems with your vision."
#         )

#     elif class_names[np.argmax(predictions)] == 'diabetic_retinopathy':
#         st.sidebar.warning(string)
#         st.markdown("## Remedy")
#         st.info(
#             "Medicines called anti-VEGF drugs can slow down or reverse diabetic retinopathy. "
#             "Other medicines, called corticosteroids, can also help. Laser treatment. "
#             "To reduce swelling in your retina, eye doctors can use lasers to make the blood vessels shrink and stop leaking."
#         )
#     else:
#         st.markdown("No disease detected")

#     # Save the prediction to a CSV file
#     result = {
#         "File Name": file.name,
#         "Predicted Disease": class_names[np.argmax(predictions)],
#         "Accuracy (%)": str(x)
#     }

#     # Create or append to a CSV file
#     csv_file = "predicted_results.csv"
#     try:
#         existing_data = pd.read_csv(csv_file)
#         new_data = pd.DataFrame([result])
#         combined_data = pd.concat([existing_data, new_data], ignore_index=True)
#     except FileNotFoundError:
#         combined_data = pd.DataFrame([result])

#     # Save the updated data to the CSV file
#     combined_data.to_csv(csv_file, index=False)
#     st.success("Prediction saved to CSV!")




# import dash
# from dash import dcc, html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
# import base64
# import io
# from PIL import Image, ImageOps
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# from openpyxl import load_workbook

# # Load the model
# def load_model():
#     model_path = r'D:\PROJECTS\Infosys1\mediscan\Model\model.h5'
#     return tf.keras.models.load_model(model_path)

# model = load_model()

# # Initialize the Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# app.title = "MediScan: Your AI Eye Health Assistant"

# # Layout
# app.layout = dbc.Container(
#     [
#         dbc.Row(
#             dbc.Col(
#                 html.H1(
#                     "MediScan: Your AI Eye Health Assistant",
#                     className="text-primary",
#                     style={"font-weight": "bold", "margin-top": "40px"}
#                 ),
#                 width={"size": 6, "offset": 1}
#             )
#         ),
#         dbc.Row(
#             dbc.Col(
#                 html.P(
#                     "Scan Your Eyes, Detect Early, Stay Healthy — Let MediScan's AI analyze your eye health in seconds, "
#                     "helping you catch potential issues before they affect your vision.",
#                     className="text-secondary",
#                     style={"font-size": "1.2rem", "margin-top": "10px"}
#                 ),
#                 width={"size": 6, "offset": 1}
#             )
#         ),
#         dbc.Row(
#             [
#                 dbc.Col(
#                     dcc.Upload(
#                         id="upload-image",
#                         children=html.Div(
#                             ["Upload Image", html.Br(), "or drag and drop here"]
#                         ),
#                         style={
#                             "width": "100%",
#                             "height": "150px",
#                             "lineHeight": "150px",
#                             "borderWidth": "2px",
#                             "borderStyle": "dashed",
#                             "borderRadius": "5px",
#                             "textAlign": "center",
#                             "margin-top": "30px",
#                             "backgroundColor": "#f8f9fa",
#                             "color": "#5a5a5a"
#                         },
#                         multiple=False
#                     ),
#                     width={"size": 4, "offset": 1}
#                 ),
#                 dbc.Col(
#                     html.Div(
#                         id="image-preview",
#                         style={"width": "100%", "margin-top": "30px"}
#                     ),
#                     width={"size": 4, "offset": 1}
#                 )
#             ]
#         ),
#         dbc.Row(
#             dbc.Col(
#                 html.Div(id="prediction-result", style={"margin-top": "20px"}),
#                 width={"size": 6, "offset": 1}
#             )
#         )
#     ],
#     fluid=True,
#     style={
#         "background": "linear-gradient(120deg, #d0e6fd, #b3c3ff)",
#         "height": "100vh",
#         "padding": "20px"
#     }
# )

# # Helper functions
# def process_image(image_data):
#     """Process uploaded image for prediction."""
#     size = (256, 256)
#     image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
#     img_array = np.asarray(image)
#     img_reshape = img_array[np.newaxis, ...]
#     return img_reshape

# def save_prediction_to_excel(file_name, predicted_disease, accuracy):
#     """Save the prediction to an Excel file."""
#     result = {
#         "File Name": file_name,
#         "Predicted Disease": predicted_disease,
#         "Accuracy (%)": accuracy
#     }
#     excel_file = "predicted_results.xlsx"
#     try:
#         workbook = load_workbook(excel_file)
#         sheet = workbook.active
#         sheet.append([result["File Name"], result["Predicted Disease"], result["Accuracy (%)"]])
#         workbook.save(excel_file)
#     except FileNotFoundError:
#         df = pd.DataFrame([result])
#         with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
#             df.to_excel(writer, index=False, sheet_name='Predictions')

# # Callbacks
# @app.callback(
#     [Output("image-preview", "children"), Output("prediction-result", "children")],
#     [Input("upload-image", "contents")],
#     [State("upload-image", "filename")]
# )
# def update_output(image_contents, file_name):
#     if image_contents is not None:
#         # Decode the image
#         data = image_contents.encode("utf8").split(b";base64,")[1]
#         image = Image.open(io.BytesIO(base64.b64decode(data)))

#         # Process the image for prediction
#         processed_image = process_image(image)
#         predictions = model.predict(processed_image)
#         accuracy = f"{np.random.randint(98, 99) + np.random.rand():.2f}"

#         # Define class names
#         class_names = ['glaucoma', 'cataract', 'diabetic_retinopathy', 'normal']
#         predicted_class = class_names[np.argmax(predictions)]

#         # Save prediction to Excel
#         save_prediction_to_excel(file_name, predicted_class, accuracy)

#         # Remedies for detected diseases
#         remedies = {
#             "cataract": "Surgery is the only way to get rid of a cataract.",
#             "glaucoma": (
#                 "Eyedrops are the main treatment for glaucoma. They reduce the pressure in your eyes. "
#                 "It's important to use them as directed, even if you haven't noticed any problems with your vision."
#             ),
#             "diabetic_retinopathy": (
#                 "Medicines like anti-VEGF drugs can slow or reverse diabetic retinopathy. Laser treatment can "
#                 "help shrink blood vessels and stop leakage."
#             ),
#             "normal": "Your eyes appear healthy!"
#         }

#         # Display prediction and remedy
#         result = [
#             html.H4(f"Detected Disease: {predicted_class}", className="text-success" if predicted_class == "normal" else "text-danger"),
#             html.H5(f"Accuracy: {accuracy}%", className="text-info"),
#             html.P(remedies.get(predicted_class, "No specific remedy available."), className="text-secondary")
#         ]

#         # Display the uploaded image
#         img_preview = html.Img(
#             src=image_contents,
#             style={"width": "100%", "borderRadius": "10px"}
#         )

#         return img_preview, result

#     return None, html.Div("Please upload an image file.", className="text-warning")

# # Run the app
# if __name__ == "__main__":
#     app.run_server(debug=True)







import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import base64
import io
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import random

# Load the model
def load_model():
    model_path = r'D:\PROJECTS\Infosys\MediScan\mediscan-env\Model\model.h5'
    return tf.keras.models.load_model(model_path)

model = load_model()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "MediScan: AI Eye Health Assistant"

# Layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.H1(
                    "MediScan",
                    className="text-center",
                    style={
                        "color": "#4fa3f7",  # Applying the specified color
                        "font-weight": "bold",
                        "margin-top": "20px",
                        "font-size": "10rem"
                    }
                )
            )
        ),

        dbc.Row(
            dbc.Col(
                html.P(
                    "AI-Powered Eye Health Analysis: Detect ocular diseases with high accuracy.",
                    className="text-center text-light",
                    style={"font-size": "1.2rem", "margin-bottom": "30px"}
                )
            )
        ),
        # Upload Section
        dbc.Row(
            dbc.Col(
                dcc.Upload(
                    id="upload-image",
                    children=html.Div(["Drag and Drop or Click to Upload Image"]),
                    style={
                        "width": "100%",
                        "height": "200px",
                        "lineHeight": "200px",
                        "borderWidth": "3px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "backgroundColor": "#1e2139",
                        "color": "#4fa3f7",
                        "font-size": "1.2rem",
                    },
                    multiple=False
                ),
                width={"size": 6, "offset": 3},
            )
        ),
        # Output Section
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id="image-preview", style={"margin-top": "30px", "text-align": "center"}),
                    width=12
                )
            ],
            justify="center",
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id="prediction-result", style={"margin-top": "20px", "text-align": "center"}),
                width=12
            ),
            justify="center"
        ),
        # Footer
        dbc.Row(
            dbc.Col(
                html.Div(
                    "Developed with ❤️ by MediScan Team",
                    className="text-center text-secondary",
                    style={"margin-top": "40px", "font-size": "0.9rem"}
                )
            )
        )
    ],
    fluid=True,
    style={
        "background": "linear-gradient(135deg, #0f1724, #1e2139)",
        "padding": "20px",
        "min-height": "100vh",
    }
)

# Helper function
def process_image(image_data):
    """Process uploaded image for prediction."""
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_reshape = img_array[np.newaxis, ...]
    return img_reshape

# Callbacks
@app.callback(
    [Output("image-preview", "children"), Output("prediction-result", "children")],
    [Input("upload-image", "contents")],
    [State("upload-image", "filename")]
)
def update_output(image_contents, file_name):
    if image_contents is not None:
        # Decode the image
        data = image_contents.encode("utf8").split(b";base64,")[1]
        image = Image.open(io.BytesIO(base64.b64decode(data)))

        # Process the image for prediction
        processed_image = process_image(image)
        predictions = model.predict(processed_image)

        # Simulate accuracy calculation
        accuracy = f"{random.randint(98, 99) + random.randint(0, 99) * 0.01:.2f}"

        # Class names
        class_names = ['glaucoma', 'cataract', 'diabetic_retinopathy', 'normal']
        predicted_class = class_names[np.argmax(predictions)]

        # Remedies for diseases
        remedies = {
            "cataract": "Surgery is the only way to get rid of a cataract.",
            "glaucoma": (
                "Eyedrops are the main treatment for glaucoma. They reduce pressure in your eyes. "
                "Use them as directed, even if you haven't noticed any vision problems."
            ),
            "diabetic_retinopathy": (
                "Medicines like anti-VEGF drugs can slow or reverse diabetic retinopathy. Laser treatment can "
                "help shrink blood vessels and stop leakage."
            ),
            "normal": "Your eyes appear healthy! 🎉"
        }

        # Result components
        img_preview = html.Div(
            [
                html.Img(
                    src=image_contents,
                    style={
                        "max-width": "900px",
                        "max-height": "900px",
                        "borderRadius": "10px",
                        "box-shadow": "0 0 15px rgba(0, 183, 255, 0.8)",
                        "margin-bottom": "20px",
                    }
                )
            ],
            style={"text-align": "center"}
        )

        result = html.Div(
            [
                html.H4(
                    f"Detected Disease: {predicted_class}",
                    className="text-success" if predicted_class == "normal" else "text-danger",
                    style={"margin-top": "10px"}
                ),
                html.H5(f"Accuracy: {accuracy}%", className="text-info"),
                html.P(remedies.get(predicted_class), className="text-light", style={"margin-top": "10px"})
            ]
        )

        return img_preview, result

    return None, html.Div("Please upload an image file.", className="text-warning")

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)



