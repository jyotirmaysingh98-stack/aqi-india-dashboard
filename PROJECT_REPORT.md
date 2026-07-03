# AQI India — Pollution Intelligence Dashboard Report (High-Visual Edition)

---

## 1. Local Interactive Dashboard Screenshot
Here is the visual mockup of our running Streamlit Web Application interface, showing the interactive panels, KPIs, geo-spatial mapping, and prediction controls:

![Streamlit App Interface Showcase](outputs/charts/dashboard_mockup.png)

---

## 2. Key Metrics Summary
The local pipeline analyzed the raw monitoring logs and structured the data into key highlights:

| Metric Indicator | Current Value | Description |
| :--- | :--- | :--- |
| **Total Stations** | **494** | Number of active monitoring stations validated across India. |
| **Mean PM2.5 Level** | **55.0 µg/m³** | Average fine particulate concentration across the dataset. |
| **Anomalies Detected** | **95 logs (3.0%)** | Outliers successfully flagged and removed using the IQR method. |
| **Severe Locations** | **2 stations (0.4%)** | Stations with a composite SPIS score greater than 75. |

---

## 3. Pollutant Concentration Distribution
This box plot displays the concentration spread for each pollutant type, excluding anomalies to prevent sensor bias:

![Pollutant Concentration Spread](outputs/charts/pollutant_distribution.png)

---

## 4. Geo-Spatial Mapping & Hotspots
The geographic map below highlights the regional distribution of the average pollution levels. Strong red points show major industrial corridors:

![Geographical Pollution Hotspots Map](outputs/charts/geo_hotspots.png)

---

## 5. Station Pollution Intelligence Score (SPIS)
The **SPIS** is our unique, patent-oriented multi-factor rating (scale 0-100) calculated using four metrics:

$$\text{SPIS} = 0.40 \times \text{AverageConcentration} + 0.25 \times \text{ExceedanceSpread} + 0.20 \times \text{AnomalyRate} + 0.15 \times \text{LogPersistence}$$

### Stations Grouped by SPIS Risk Bands:
The distribution bar chart displays the count of monitoring stations classified under each risk level:

![SPIS Risk Band Classification Counts](outputs/charts/spis_risk_bands.png)

### Top 5 Most Critical Stations (Highest SPIS):
| Monitoring Station | City | State | SPIS Score (0-100) | Risk Band |
| :--- | :--- | :--- | :---: | :--- |
| **Fertilizer Township, Rourkela - OSPCB** | Rourkela | Odisha | **92.5** | **Severe** |
| **Sector-1, Rourkela - OSPCB** | Rourkela | Odisha | **82.3** | **Severe** |
| **Balkum, Thane - MPCB** | Thane | Maharashtra | **73.4** | **High** |
| **Karve Road, Pune - MPCB** | Pune | Maharashtra | **71.2** | **High** |
| **Pimpri, Chinchwad - MPCB** | Pimpri-Chinchwad | Maharashtra | **69.8** | **High** |

---

## 6. City Rankings (Top 15 Most Polluted)
The bar chart below ranks the top cities in India based on their mean pollutant concentrations:

![Top Cities by Mean Pollutant Concentration](outputs/charts/top_cities.png)

---

## 7. Machine Learning Classification Performance
We trained a **Random Forest Classifier** to assess the interpretability of our risk bands. The model achieved an outstanding validation accuracy of **97%**.

### Precision & Recall Classification Report:
| Target Risk Class | Precision Accuracy | Recall Rate | F1-Score | Validated Instances |
| :--- | :---: | :---: | :---: | :---: |
| **Low** | 98% | 95% | 96% | 42 |
| **Moderate** | 96% | 99% | 97% | 77 |
| **High** | 100% | 75% | 86% | 4 |
| **Severe** | 100% | 100% | 100% | 1 |
| **Weighted Average** | **97%** | **97%** | **97%** | **124** |

The feature importance weights verify that **average pollutant concentration (`mean_avg`)** and **variance spread (`mean_spread`)** are the main driving features of the SPIS classification:

![Model Feature Importances Chart](outputs/charts/risk_band_feature_importance.png)

---

## 8. Conclusion
This highly visual report highlights that the pipeline is fully operational. The Streamlit Web App allows users to dynamically edit records, inject synthetic data via the augmentation engine, and retrain the machine learning model in real-time, instantly updating the SPIS maps and accuracy tables.
