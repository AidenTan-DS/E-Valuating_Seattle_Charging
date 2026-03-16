# Evaluating Seattle Charging App

## Welcome
- This application helps you analyze and evaluate electric vehicle charging station placement across Seattle using interactive maps, statistics fields, and machine learning. 

## What Is Available
This document will guide users to use our app and:

- Understand traffic patterns and EV infrastructure
- Evaluate existing EV charging station placement quality
- View the predictions of future development of charging infrastructure

## Before Use
- Install all the required packages, information and instructions are in *docs/environment_setup.md*
- Activate the ev_env virtual environment with conda activate ev_env from the git repository command line.

## Walk Through

### Running the App
Once the virtual environment is activated, run in terminal
```bash
streamlit run interactive_map/app_v2.py
```
This should autmatically take you to a new browser where the our app will be locally hosted. 

## Tab 1 — Locations

The first tab you will be brought to is the **Locations** tab.

Here, the map on the left is an interactive choropleth map showing all Seattle ZIP codes. Each ZIP code is color‑coded by **average daily traffic flow** based on the legend on the right.

To investigate the traffic flow of a specific ZIP code:

1. **Right‑click** on the ZIP code boundary in the map on the left or enter Zip Code on drop down menu.
2. A **close‑up detailed map** of that ZIP code will populate on the right.

This detailed view lets you see:

- Established EV chargers in that area
- Traffic flow on neighboring streets (also color‑coded for easy interpretation)

This makes it simple to explore high‑traffic areas and understand where existing charging infrastructure is relative to traffic demand.

## Tab 2 — Evaluation

The **Evaluation** tab lets you customize how charging station quality is assessed across Seattle using interactive weight sliders.

In this tab:

- You’ll see two **input weight sliders** with values from **0 to 10**.
- These sliders allow you to assign your own **desired importance (weights)** to:
  - **Traffic**
  - **demand gap**  
  - **Population density**  
  for each ZIP code.

As you adjust the sliders:

- Each EV charging station on the map is **evaluated** based on your selected weights.
- The **evaluation score** for each station combines the traffic demand gap and population density according to the weights you set and is showcased in the statistics table to the right.
- The **color of the EV station markers** on the map **updates dynamically** to reflect their evaluation score — stations with higher scores will be visually distinguished from those with lower scores.

This interactive weighting system gives developers and planners flexibility to explore different prioritization scenarios, customizing how demand and population influence station evaluation in real time.

## Tab 3 — Predictions

The **Predictions** tab displays the output from our machine learning model, which has been trained on existing charging station data and relevant geographic features.

In this tab:

- You’ll see a map visualization showing **predicted EV charging station locations** based on patterns learned from the stations that are already established.
- The slider allows the user to filter the probability of an ev station being developed based on the model.
- The model uses historical and geographic inputs to estimate where new charging stations would be most beneficial or likely to be placed in the future.  

This tab gives planners and analysts a forward‑looking view of charging infrastructure needs, helping inform strategic decisions and infrastructure planning based on model projections.