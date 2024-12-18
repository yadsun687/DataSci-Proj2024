import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
st.set_page_config(page_title="Scopus dataset VIZ" ,layout="wide")

st.title("Visualization of Scopus Dataset")

@st.cache_data
def load_data():
    data = pd.read_parquet('data/viz_data.parquet.gzip')
    asjc_df = pd.read_csv('data/ASJC_cat.csv')
    coord_df = pd.read_csv("data/coordinate_country.csv" , index_col=0)
    return data , asjc_df , coord_df

scopus_data , asjc_df , coord_df = load_data()
asjc_df.set_index('Code',inplace=True)
asjc_dict = asjc_df["ASJC category"].to_dict()
asjc_cat_dict = asjc_df["ASJC category"].to_dict()
asjc_subj_dict = asjc_df["Subject area"].to_dict()


#=====================Most publish stats=================================
st.header(f'Articles in subject area' , divider="red")

most_subj_df = scopus_data.copy()
most_subj_df = most_subj_df.explode(column="ASJC_code") #explode list of subject code into multiple rows
most_subj_df = most_subj_df["ASJC_code"].value_counts().rename_axis('ASJC_code').reset_index(name='counts')
most_subj_df["Category"] = most_subj_df["ASJC_code"].astype(int).map(asjc_cat_dict)
most_subj_df["Subject_area"] = most_subj_df["ASJC_code"].astype(int).map(asjc_subj_dict)

st.subheader("Pie Chart")
_ ,middle , _ = st.columns([2,5,2])
with middle:    
    pie_chart = px.pie(most_subj_df ,values="counts", color_discrete_sequence=px.colors.sequential.RdBu,color="counts" , names='Subject_area' , title='Published articles' ,height=600)
    st.plotly_chart(pie_chart)

st.subheader("Treemap")
tree_map = px.treemap(most_subj_df, path=["Subject_area" , "Category"] , color_continuous_scale=px.colors.sequential.RdBu_r , color="counts" , values='counts' , height=1000)
tree_map.data[0].textinfo = 'label+value    '
tree_map.update_layout(
font=dict(

    size=24
))
st.plotly_chart(tree_map)


# tree_map.update_traces(textposition='')
    

#=====================CiteRef stats=================================
st.header(f'Citational/Refererence Statistics' , divider="red")

#drop null row & cast to int
month_stats_df = scopus_data.copy()
month_stats_df = scopus_data.dropna(subset=["ref_count" , "citedby_count"],axis=0 , how="any")
month_stats_df["ref_count"] = month_stats_df["ref_count"].astype(int)
month_stats_df["citedby_count"] = month_stats_df["citedby_count"].astype(int)

month_stats_df["year"] = month_stats_df["delivered_date"].dt.year
month_stats_df["month"] = month_stats_df["delivered_date"].dt.month

max_ref = month_stats_df.groupby(by=["year","month"])["ref_count"].sum().max()
max_cited = month_stats_df.groupby(by=["year","month"])["citedby_count"].sum().max()

col1, col2 = st.columns(2)
with col1:
    st.metric("Total citations", month_stats_df["citedby_count"].sum())
with col2:
    st.metric("Total references", month_stats_df["ref_count"].sum())
    
col1 , _ , _= st.columns(3)
with col1:
    count_metric = st.selectbox("choose metric" , options=["Citation" , "Reference"])
    metric_map = {"Citation" : "citedby_count" , "Reference" : "ref_count"}

fig = px.histogram(month_stats_df,
                   x="month",
                   y = metric_map[count_metric] , 
                   animation_frame="year" ,
                   title = f"Number of {count_metric} used each month" , 
                   category_orders={"year" : [2019,2020,2021,2022,2023]},
                   range_x=[1,12],
                   range_y=[0,max_cited if (count_metric == "Citation") else max_ref],
                   labels={ metric_map[count_metric] : count_metric}
                   )
fig.update_layout(transition = {"duration" : 100} )
fig.update_xaxes(dtick= "M1" , tickformat="%B")
st.plotly_chart(fig)


#=====================Funding stats=================================
st.header("How many journals are funded?",divider="red")

funded_sbj_df = scopus_data.copy()
funded_sbj_df = funded_sbj_df.explode(column="ASJC_code") #list of subject code to multiple rows
funded_sbj_df:pd.DataFrame = funded_sbj_df.groupby("ASJC_code",as_index=False)["is_funding"].sum()
funded_sbj_df["Category"] = funded_sbj_df["ASJC_code"].astype(int).map(asjc_cat_dict)
funded_sbj_df["Subject_area"] = funded_sbj_df["ASJC_code"].astype(int).map(asjc_subj_dict)

# get unique color map
unique_sbj_area = funded_sbj_df["Subject_area"].unique()
total_sbj_area = len(unique_sbj_area)
colormap = plt.get_cmap('hsv')
cluster_colors = {cluster: [int(x*255) for x in colormap(i/(total_sbj_area))[:3]]
 for i, cluster in enumerate(unique_sbj_area)}

funded_sbj_df["color"] = funded_sbj_df["Subject_area"].map(cluster_colors)

st.subheader("Total funding in each category")
cols = st.columns(funded_sbj_df["Subject_area"].nunique())
for i, x in enumerate(cols):
    x.metric(
        funded_sbj_df["Subject_area"].unique()[i] , 
        funded_sbj_df.groupby("Subject_area")["is_funding"].sum().loc[funded_sbj_df["Subject_area"].unique()[i]])

horizontal_bar_g = px.bar(funded_sbj_df ,title="Total number of journal which has been funded", x="is_funding",y="Category" , color="Subject_area" , labels={"is_funding" : "Total funding"})
st.plotly_chart(horizontal_bar_g)
st.caption("Some category needed to be zoomed to be visible")



#=====================Affiliation map=================================
st.header("Affiliations around the world" , divider="red")
st.write("Same articles might appear in many place (because it have more than 1 affiliations)")

coord_df = coord_df.set_index("country_name" , drop = True)
scopus_with_coord = scopus_data.copy()
scopus_with_coord = scopus_with_coord.explode("affiliations_country" , ignore_index=True)
scopus_with_coord = scopus_with_coord.dropna(how="any" , subset=["affiliations_country"]) 
scopus_with_coord["lng"] = scopus_with_coord["affiliations_country"].map(coord_df["longitude"]).astype("Float64")
scopus_with_coord["lat"] = scopus_with_coord["affiliations_country"].map(coord_df["latitude"]).astype("Float64")

# highest total affiliation country 
top_country = scopus_with_coord["affiliations_country"].value_counts(sort=True).head(15)
    
scatter_layer = pdk.Layer("ScatterplotLayer", scopus_with_coord[["lng" , "lat" , "title" , "main_author" , "affiliations_country"]] ,
                  get_position = ["lng","lat"] ,
                  get_radius=10000 , 
                  get_fill_color=[255, 140, 0] , 
                  opacity=0.5 ,
                  pickable =True
                  )
viewState = pdk.ViewState(latitude=scopus_with_coord["lat"].mean(),
                          longitude=scopus_with_coord["lng"].mean() , 
                          zoom=1
                          )
scatter_rendered = pdk.Deck(layers=[scatter_layer] , 
                    initial_view_state=viewState , 
                    map_style=pdk.map_styles.MAPBOX_DARK , 
                    tooltip={"text": "Title: {title}\nMain author: {main_author}\nLocation: {affiliations_country}"}
                    )
st.pydeck_chart(scatter_rendered)
st.subheader("Top 15 with most affiliations")
st.dataframe(top_country , width=500)
#==================================================================