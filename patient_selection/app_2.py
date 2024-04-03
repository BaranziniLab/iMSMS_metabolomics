import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load the data
def load_data():
    patient_feature_df = pd.read_csv('../../wetlab/data/patient_selection/patient_profile.csv')  
    patient_feature_array = patient_feature_df.drop(['CLIENT_SAMPLE_ID', 'gARMSS', 'armms_category'], axis=1).to_numpy()
    return patient_feature_df, patient_feature_array

df, patient_feature_array = load_data()

# Define colors for armms_category
color_map = {'Lower ARMSS': 'green', 'Higher ARMSS': 'red'}

# Plot
fig = px.scatter(df, x='tsne1', y='tsne2', color='armms_category', color_discrete_map=color_map,
                 hover_data={'tsne1': False, 'tsne2': False, 'CLIENT_SAMPLE_ID': True, 'gARMSS': True})

# Update legend title
fig.update_layout(legend_title_text='ARMSS Category')

# Update marker size
fig.update_traces(marker=dict(size=10))  # Adjust the size of the dots here

# Update legend labels
fig.update_traces(
    name="Lower ARMSS",
    selector=dict(name="green")
)
fig.update_traces(
    name="Higher ARMSS",
    selector=dict(name="red")
)

# Update figure size
fig.update_layout(height=600, width=800)  # Adjust the height and width of the figure here


# Increase the size of x and y axis labels
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

fig.update_layout(xaxis=dict(range=[-30, 28]), yaxis=dict(range=[-35, 28]))

# Add functionality to compute L1 norm and display top n largest CLIENT_SAMPLE_ID values
st.sidebar.markdown("# Select Options:")
selected_client_sample_id = st.sidebar.selectbox('Select CLIENT_SAMPLE_ID:', [None] + df['CLIENT_SAMPLE_ID'].tolist())

if selected_client_sample_id:
    num_top_l1_norms = st.sidebar.slider('Number of Top L1 Norms:', 1, 100, 10)

    selected_index = df.index[df['CLIENT_SAMPLE_ID'] == selected_client_sample_id][0]
    
    
    selected_vector = patient_feature_array[selected_index]
    l1_norms = np.linalg.norm(patient_feature_array - selected_vector.reshape(1, -1), ord=1, axis=1)
    top_indices = np.argsort(l1_norms)[-num_top_l1_norms:][::-1]
    top_client_sample_ids = df.loc[top_indices, 'CLIENT_SAMPLE_ID'].tolist()
    armss_category = df.loc[top_indices, 'armms_category'].tolist() 
    
    # Highlight selected points on the scatter plot
    highlighted_points = df.iloc[top_indices]
    fig = px.scatter(highlighted_points, x='tsne1', y='tsne2', color='armms_category', color_discrete_map=color_map,
                     hover_data={'tsne1': False, 'tsne2': False, 'CLIENT_SAMPLE_ID': True, 'gARMSS': True})
    # Update legend title
    fig.update_layout(legend_title_text='ARMSS Category')

    # Update marker size
    fig.update_traces(marker=dict(size=10))  # Adjust the size of the dots here

    # Update legend labels
    fig.update_traces(
        name="Lower ARMSS",
        selector=dict(name="green")
    )
    fig.update_traces(
        name="Higher ARMSS",
        selector=dict(name="red")
    )

    # Update figure size
    fig.update_layout(height=600, width=800)  # Adjust the height and width of the figure here

    # Increase the size of x and y axis labels
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
    fig.update_layout(xaxis=dict(range=[-30, 28]), yaxis=dict(range=[-35, 28]))

#     # Color other points in transparent gray
#     grayed_out_points = df.drop(index=top_indices)
#     grayed_out_fig = px.scatter(grayed_out_points, x='tsne1', y='tsne2', color='armms_category').data[0]
#     grayed_out_fig.update(marker=dict(color='gray'), opacity=0.3)
#     fig.add_trace(grayed_out_fig)

    # Display top n largest CLIENT_SAMPLE_ID values as a table
    st.write("## Top", num_top_l1_norms, "Largest L1 Norms:")
    st.write(f'Selected ID = {selected_client_sample_id}')
    selected_category = df.iloc[selected_index]['armms_category']
    st.write(f'Selected ID Cateogry = {selected_category}')
    st.table(pd.DataFrame({'CLIENT_SAMPLE_ID': top_client_sample_ids, 'Category':armss_category}))
else:
    st.write("## Top 10 Largest L1 Norms:")
    st.write("Please select a CLIENT_SAMPLE_ID from the sidebar.")

# Display the plot
st.plotly_chart(fig, use_container_width=True)
