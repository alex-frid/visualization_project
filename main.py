import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import plotly.express as px
import plotly.graph_objects as go


# Setup
st.set_page_config(page_title="Student Performance")
st.title("📊 Student Performance insights")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Demographic Analysis of Dropout Rates", "Academic Performance and Prior Achievements", "Impact of Parental Education on Student Performance"])

@st.cache_data
def load_data():
    """Fetches and caches the dataset."""
    dataset = fetch_ucirepo(id=697)
    X = dataset.data.features
    y = dataset.data.targets
    return pd.concat([X, y], axis=1)

df = load_data()

def visualization1(category):
    """Displays a bar chart comparing dropouts vs. non-dropouts."""
    vis1 = df.copy()
    vis1['Age at enrollment'] = pd.cut(
        vis1['Age at enrollment'],
        bins=[0, 20, 25, 30, 100],
        labels=['<20', '20-25', '25-30', '>30']
    )

    vis1['Gender'] = vis1['Gender'].map({1: 'Male', 0: 'Female'})
    vis1['Marital Status'] = vis1['Marital Status'].map({
        1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Separated',
        5: 'Married', 6: 'Separated'
    })
    vis1['Nationality'] = vis1['Nacionality'].apply(lambda x: 'Portuguese' if x == 1 else 'Other')

    status_counts = vis1.groupby([category, 'Target']).size().unstack(fill_value=0)
    total_counts = status_counts.sum(axis=1)  # Total counts for each category to calculate percentages

    # Prepare data for Plotly
    x_labels = status_counts.index.astype(str)

    # Create traces for Dropouts and Non-Dropouts
    fig = go.Figure()
    if 'Dropout' in status_counts.columns:
        fig.add_trace(go.Bar(
            x=x_labels,
            y=status_counts['Dropout'],
            name='Dropouts',
            marker_color='blue',
            customdata=[
                f"Count: {val}<br>Percentage: {(val / total_counts[i]) * 100:.1f}%" if total_counts[i] > 0 else "" for
                i, val in enumerate(status_counts['Dropout'])],
            hovertemplate="%{customdata}",
        ))

    non_dropout_columns = [col for col in ['Graduate', 'Enrolled'] if col in status_counts.columns]
    if non_dropout_columns:
        non_dropout_values = status_counts[non_dropout_columns].sum(axis=1)
        fig.add_trace(go.Bar(
            x=x_labels,
            y=non_dropout_values,
            name='Non-Dropouts',
            marker_color='orange',
            customdata=[
                f"Count: {val}<br>Percentage: {(val / total_counts[i]) * 100:.1f}%" if total_counts[i] > 0 else "" for
                i, val in enumerate(non_dropout_values)],
            hovertemplate="%{customdata}",
        ))

    # Update layout
    fig.update_layout(
        title=f'Distribution by {category}',
        xaxis=dict(title=category),
        yaxis=dict(title='Number of Students'),
        barmode='group',
        legend_title='Target',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Show the figure
    st.plotly_chart(fig)

def visualization2(options):
    vis2 = df.copy()
    # Define columns to normalize
    columns_to_normalize = [
        'Previous qualification (grade)',
        'Admission grade',
    ]

    # Desired mean and standard deviation for normalization
    target_mean = 10
    target_std = 2

    # Perform normalization
    for col in columns_to_normalize:
        original_mean = vis2[col].mean()
        original_std = vis2[col].std()

        # Calculate z-score and adjust to the new mean and standard deviation
        vis2[col + ' normalized'] = ((vis2[col] - original_mean) / original_std) * target_std + target_mean

    # Rename normalized columns for better readability in graphs
    normalized_columns_rename = {
        'Previous qualification (grade) normalized': 'Previous qual. grade',
        'Admission grade normalized': 'Admission grade ',
        'Curricular units 1st sem (grade)': '1st sem grade',
        'Curricular units 2nd sem (grade)': '2nd sem grade',
    }

    vis2.rename(columns=normalized_columns_rename, inplace=True)

    # Consolidate 'Graduate' and 'Enrolled' under 'Non-Dropout' only during visualization
    vis2['Target'] = vis2['Target'].replace({'Graduate': 'Non-Dropout', 'Enrolled': 'Non-Dropout'})

    # Define a new colorblind-friendly palette
    custom_palette = {
        'Dropout': '#CC79A7',  # Purple
        'Non-Dropout': '#F0E442'  # Yellow
    }

    # Scatter plot matrix with color coding for 'Target'
    scatter_matrix_colored = sns.pairplot(
        vis2[options],
        diag_kind='kde',
        # hue='Target',
        height=2.5
    )
    # Adding a title to the scatter plot matrix
    scatter_matrix_colored.fig.suptitle('Scatter Plot Matrix', y=1.02)

    st.pyplot(scatter_matrix_colored)

def visualization3(parent):
    """Displays educational background impact using radar charts."""
    vis3 = df.copy()
    # mother educational mapping
    education_mapping_m = {
        1: 'Secondary Education',
        2: 'Higher Education',
        3: 'Higher Education',
        4: 'Higher Education',
        5: 'Higher Education',
        6: 'Other Education',
        9: 'Secondary Education',
        10: 'Secondary Education',
        11: 'Primary Education',
        12: 'Secondary Education',
        14: 'Secondary Education',
        18: 'Secondary Education',
        19: 'Primary Education',
        22: 'Other Education',
        26: 'Primary Education',
        27: 'Secondary Education',
        29: 'Primary Education',
        30: 'Primary Education',
        34: 'Uneducated',
        35: 'Uneducated',
        36: 'Uneducated',
        37: 'Primary Education',
        38: 'Primary Education',
        39: 'Higher Education',
        40: 'Higher Education',
        41: 'Higher Education',
        42: 'Higher Education',
        43: 'Higher Education',
        44: 'Higher Education'
    }

    # father educational mapping
    education_mapping_f = {
        1: 'Secondary Education',
        2: 'Higher Education',
        3: 'Higher Education',
        4: 'Higher Education',
        5: 'Higher Education',
        6: 'Other Education',
        9: 'Secondary Education',
        10: 'Secondary Education',
        11: 'Primary Education',
        12: 'Secondary Education',
        13: 'Secondary Education',
        14: 'Secondary Education',
        18: 'Secondary Education',
        19: 'Primary Education',
        20: 'Secondary Education',
        22: 'Other Education',
        25: 'Secondary Education',
        26: 'Primary Education',
        27: 'Secondary Education',
        29: 'Primary Education',
        30: 'Primary Education',
        31: 'Secondary Education',
        33: 'Secondary Education',
        34: 'Uneducated',
        35: 'Uneducated',
        36: 'Uneducated',
        37: 'Primary Education',
        38: 'Primary Education',
        39: 'Higher Education',
        40: 'Higher Education',
        41: 'Higher Education',
        42: 'Higher Education',
        43: 'Higher Education',
        44: 'Higher Education'
    }

    vis3["EducationLevel Father"] = vis3["Father's qualification"].map(education_mapping_f)
    vis3["EducationLevel Mother"] = vis3["Mother's qualification"].map(education_mapping_m)


    def plot_grade_boxplot(education_column, sem1_column, sem2_column, title):
        """
        Plots a box plot comparing the distribution of grades for Semester 1 and Semester 2
        across different education levels.

        Parameters:
            education_column (str): The name of the education level column.
            sem1_column (str): The name of the first semester grades column.
            sem2_column (str): The name of the second semester grades column.
            title (str): The title of the chart.
        """
        # Prepare data
        vis3_long = vis3.melt(id_vars=[education_column],
                              value_vars=[sem1_column, sem2_column],
                              var_name="Semester",
                              value_name="Grade")

        # Rename semesters for clarity
        vis3_long["Semester"] = vis3_long["Semester"].replace({
            sem1_column: "Semester 1",
            sem2_column: "Semester 2"
        })

        # Reorder education levels as specified
        education_order = ['Uneducated', 'Primary Education', 'Secondary Education', 'Higher Education',
                           'Other Education']

        # Create box plot
        fig = px.box(vis3_long,
                     x=education_column,
                     y="Grade",
                     color="Semester",
                     title=title,
                     labels={education_column: "Education Level", "Grade": "Student Grades"},
                     color_discrete_map={"Semester 1": "#005AB5", "Semester 2": "#EC90C3"},
                     category_orders={education_column: education_order})  # Blue and Purple

        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(vis3_long[education_column].unique()) - 0.5,  # Covers the entire x-axis
            y0=10, y1=10,
            line=dict(color="red", width=2, dash="dash"),  # Red dashed line
        )

        # Add an invisible trace to make the dotted line appear in the legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible
            mode="lines",
            line=dict(color="red", width=2, dash="dot"),
            name="Pass Threshold"
        ))

        # Update layout
        fig.update_layout(xaxis_title="Education Level",
                          yaxis_title="Grade",
                          boxmode="group",  # Grouped boxes
                          template="plotly_white",
                          yaxis = dict(
                          tickmode="array",
                          tickvals=[0, 5, 10, 15, 20],
                          range=[0, 20]),
                          xaxis=dict(tickangle=0),
                          bargap=0.5
                          )



        # Display in Streamlit
        st.plotly_chart(fig)

    if parent == "Father's Education level":
        plot_grade_boxplot(
            education_column='EducationLevel Father',
            sem1_column='Curricular units 1st sem (grade)',
            sem2_column='Curricular units 2nd sem (grade)',
            title="Grade Distribution by Father's Education Level"
        )
    elif parent == "Mother's Education level":
        plot_grade_boxplot(
            education_column='EducationLevel Mother',
            sem1_column='Curricular units 1st sem (grade)',
            sem2_column='Curricular units 2nd sem (grade)',
            title="Grade Distribution by Mother's Education Level"
        )


if page == "Demographic Analysis of Dropout Rates":
    st.write("This chart highlights dropout rates across various demographic groups.")
    col1, col2 = st.columns([1.5, 3])  # Adjust column width ratios
    with col1:
        category = st.selectbox("Select a Demographic Feature:",
                                ['Gender', 'Marital Status', 'Age at enrollment', 'Nationality'])

    visualization1(category)
    visualization1(category)

elif page == "Academic Performance and Prior Achievements":
    st.write("This matrix scatter plot visualizes relationships between prior achievements, academic performance, or both.")
    options = st.multiselect("Choose Performances to Explore:",['Previous qual. grade', 'Admission grade ', '1st sem grade', '2nd sem grade'],['Previous qual. grade','Admission grade '])

    if len(options) < 2:
        st.warning("Please select at least two options.")
    else:
        visualization2(options)

elif page == "Impact of Parental Education on Student Performance":
    st.write("This grouped bar chart illustrates how parental education levels influence student success during their first and second semesters.")
    parent = st.radio("Select Parent:", ["Father's Education level", "Mother's Education level"])
    visualization3(parent)