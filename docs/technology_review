## What is your app?
We aim to identify where Seattle could benefit from additional EV charging stations, evaluate how well the city’s existing charging infrastructure meets demand, and compare current station locations with our data‑driven recommended sites using an efficiency metric. Our project will culminate in an interactive dashboard that lets users explore both existing and suggested charging locations, visualize demand based on traffic, crime, and demographic factors, and apply customizable criteria to refine recommendations. Final deliverables include maps, efficiency visualizations, and actionable insights for optimal future charger placement.leverage this to suggest more development in certain areas.

## What use case are you trying to solve?
We are addressing the use case of visualizing EV ownership and charger infrastructure in Seattle by building an interactive map that shows EV registration data and existing charger locations, broken down by ZIP code.

## What about it needs a Python library?
Since our project relies on geographic coordinates (latitude, longitude) and ZIP code boundaries to represent EV registrations and charger locations, a geographic visualization library is essential to translate that spatial data into a visual, interactive map. Maps provide the geographic context needed to recognize spatial patterns, clusters, and gaps that are not easily discernible in tables or charts alone, and they enable users to explore and interpret the data through zooming, panning, and interactive layers. To create these maps, we used the geopandas python library to wrangle geographic coordinates and zip code areas from the dataframes we created, and used the streamlit library to display our dashboard as a web application. As such, when evaluating how each library performs, we will consider their ease of use, range of functionalities, robustness, and their compatibility with our aforementioned libraries.

## Library 1: Folium
- Author: Ross Kirsling
- Summary: Folium is an open-source Python library designed for the creation of interactive, web-based maps that support the visualization and exploration of spatial data. It functions as a Python interface to the JavaScript mapping library Leaflet, enabling users to generate dynamic maps without requiring direct knowledge of front-end web development technologies.

## Library 2: Plotly
- Author: Plotly Inc 
- Summary: Plotly is a generalized open-source Python library that is designed to allow users to explore, analyze, and visualize data through dynamic, web-based charts and plots. It is built on top of plotly.js and uses a tile-based mapping engine from a fork of Mapbox GL JS called Maplibre GL JS. To visualize our geographic data, we used px.choropleth_mapbox, a tile-based mapping function within Plotly Express, a high-level API for the plotly library that allowed plotly.graph_objects .Figure instances to be created to display ZIP code data on interactive map tiles.

## Plotly & Folium: 

Plotly and Folium are both tools that can be used for creating interactive visualizations of geographic data. Plotly is a general tool that’s able to create many different kinds of visualizations, while Folium is a dedicated tool for creating interactive geographic maps. 

Comparing the two, Folium’s focus as a mapping library means that it’s able to cover more geographic data needs compared to Plotly. The latter also lacks some native geopandas integrations, so it requires additional explicit commands for geographic data conversions to be rendered correctly. In particular, it lacks CRS (Coordinated Reference System) awareness, and requires a command like “gdf.to_crs(epsg=4326)” to be run prior to plotting a graph. Additionally, the geometry of a geopandas dataframe needs to be explicitly extracted and loaded into a separate dictionary using the command “json.loads(gdf.to_json())”.  Folium also has an edge in terms of accessibility, as it is able to load in data from TopoJSON files in addition to GeoJSON files. However, Plotly has an edge in streamlit compatibility since Folium requires a streamlit function (st_folium) to be imported for its maps to be interactive, while Plotly has native streamlit integration and doesn’t require such support. Moreover, when creating choropleth plots, Plotly’s syntax is simpler relative to Folium. 

When comparing the status of the GitHub repositories, plotly’s generalized nature means that it has many more users and contributors compared to folium (458k users and 275 contributors compared to 64.3k users and 166 contributors), but it also has ten times the number of  open Issues (716) reported compared to folium (71). 

## Library Chosen: Plotly
For this application, we decided to use the Plotly library to create an interactive map for our use case. Plotly is more compatible with Streamlit, allowing for richer interactivity and scalability. We aim to display additional information for the user, like population, income, and EV registration and Plotly is better suited for this function than Folium. Streamlit includes built in functions that directly render Plotly figures and allows them to update dynamically with user inputs, making it easy to build our responsive map. In contrast, Folium renders HTML/leaflet outputs which can limit reactivity and require additional workarounds to work with streamlit. 
