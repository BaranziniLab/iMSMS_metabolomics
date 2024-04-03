import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


patient_profile = pd.read_csv('../../wetlab/data/patient_selection/patient_profile.csv')
patient_feature_array = patient_profile.drop(['CLIENT_SAMPLE_ID', 'gARMSS', 'armms_category'], axis=1).to_numpy()



# Define colors for armms_category
color_map = {'Lower ARMSS': 'green', 'Higher ARMSS': 'red'}

# Plot
fig = px.scatter(patient_profile, x='tsne1', y='tsne2', color='armms_category', color_discrete_map=color_map,
                 hover_data={'tsne1': False, 'tsne2': False, 'CLIENT_SAMPLE_ID': True, 'gARMSS': True}
                )

# Update legend names
fig.for_each_trace(lambda t: t.update(name=color_map[t.name]))

# Update legend title
fig.update_layout(legend_title_text='ARMSS Category')

fig.update_layout(height=500, width=700)
fig.update_traces(marker=dict(size=10))
fig.update_traces(
    name = "Lower ARMSS",
    selector=dict(name="green")
)
fig.update_traces(
    name = "Higher ARMSS",
    selector=dict(name="red")
)
fig.update_layout(
    xaxis=dict(
        title=dict(
            font=dict(size=20)  # Adjust the font size of the x axis label here
        )
    ),
    yaxis=dict(
        title=dict(
            font=dict(size=20)  # Adjust the font size of the y axis label here
        )
    )
)
# Display the plot
st.plotly_chart(fig)


# def plot_scatter(df):
#     fig = px.scatter(df, x='tsne1', y='tsne2', color='armms_category',
#                      color_discrete_map={1: 'red', 0: 'green'},
#                      hover_data={'tsne1': False, 'tsne2': False,
#                                  'CLIENT_SAMPLE_ID': True, 'gARMSS': True},
#                      labels={'CLIENT_SAMPLE_ID': 'Client Sample ID', 'gARMSS': 'gARMSS'})
#     fig.update_traces(marker=dict(size=10))
#     fig.update_layout(title='tSNE Plot',
#                       xaxis_title='tSNE 1',
#                       yaxis_title='tSNE 2',
#                       legend_title='ARMSS Category',
#                       )
#     return fig


# def plot_scatter(df):
#     fig = px.scatter(df, x='tsne1', y='tsne2', color='armms_category',
#                      color_discrete_map={1: 'red', 0: 'green'},
#                      hover_data={'tsne1': False, 'tsne2': False,
#                                  'CLIENT_SAMPLE_ID': True, 'gARMSS': True},
#                      labels={'CLIENT_SAMPLE_ID': 'Client Sample ID', 'gARMSS': 'gARMSS'})
#     fig.update_traces(marker=dict(size=10))
#     fig.update_layout(title='tSNE Plot',
#                       xaxis_title='tSNE 1',
#                       yaxis_title='tSNE 2')
#     # Add legend
#     fig.update_layout(legend=dict(
#         title="ARMSS Category",
#         items=[
#             dict(label="Higher gARMSS", marker=dict(color='red', size=10)),
#             dict(label="Lower gARMSS", marker=dict(color='green', size=10)),
#         ]
#     ))
#     return fig


# def main():
#     st.title("tSNE Plot of patient profile")
#     st.plotly_chart(plot_scatter(patient_profile))

# if __name__ == "__main__":
#     main()