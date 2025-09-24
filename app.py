
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import plotly.graph_objects as go
from scipy import optimize
from numpy.polynomial import Polynomial

st.set_page_config(layout="wide", page_title="Latin Countries Historical Regression Explorer")

st.title("Latin Countries Historical Regression Explorer")

# --- Configuration: chosen candidate countries (wealthiest Latin countries by GDP per capita typically include Panama, Uruguay, Chile, Costa Rica, Argentina).
DEFAULT_COUNTRIES = ["Chile", "Uruguay", "Panama"]

# Map user-facing categories to Our World in Data grapher slugs and units
INDICATORS = {
    "Population": {"slug":"population", "unit":"people", "desc":"Total population"},
    "Unemployment rate": {"slug":"unemployment_total", "unit":"%","desc":"Unemployment rate (percent) — availability varies by country"},
    "Education levels from 0-25": {"slug":"mean-years-of-schooling", "unit":"index (0-25)","desc":"Mean years of schooling scaled to 0-25 (25 highest)"},
    "Life expectancy": {"slug":"life_expectancy", "unit":"years","desc":"Life expectancy at birth (years)"},
    "Average wealth": {"slug":"gdp_per_capita", "unit":"USD","desc":"GDP per capita (current US$) — proxy for average wealth"},
    "Average income": {"slug":"gdp_per_capita", "unit":"USD","desc":"GDP per capita (current US$) — proxy for average income"},
    "Birth rate": {"slug":"crude_birth_rate", "unit":"births per 1000 people","desc":"Crude birth rate (per 1,000 people)"},
    "Immigration out of the country": {"slug":"net_migration", "unit":"net migrants per year","desc":"Net migration (positive = net inflow, negative = net outflow)"},
    "Murder Rate": {"slug":"intentional-homicides", "unit":"deaths per 100,000 people","desc":"Intentional homicide rate (per 100k people)"}
}

# Note: OWID slugs use hyphens in some names; adjust known variants
SLUG_FIXES = {
    "unemployment_total":"unemployment_total", # may be absent for some countries
    "mean-years-of-schooling":"mean-years-of-schooling",
    "life_expectancy":"life_expectancy",
    "gdp_per_capita":"gdp_per_capita",
    "crude_birth_rate":"crude-birth-rate",
    "net_migration":"net-migration",
    "intentional-homicides":"intentional-homicides",
    "population":"population"
}

# Helper to build OWID grapher CSV URLs
def get_owid_csv(slug, countries, start=1950, end=None):
    if end is None:
        end = datetime.now().year
    # OWID grapher accepts ?country= and &time=YYYY..YYYY for wide CSVs
    cs = ",".join(countries)
    # Some slugs have underscores in our mapping but OWID filenames use hyphens; try both.
    slug_try = slug.replace("_", "-")
    url = f"https://ourworldindata.org/grapher/{slug_try}.csv?country={cs}&time={start}..{end}"
    return url

# Sidebar controls
st.sidebar.header("Controls")
countries = st.sidebar.multiselect("Select countries (up to 3)", DEFAULT_COUNTRIES, default=DEFAULT_COUNTRIES, help="Choose up to three countries to display. App will try to fetch historical data (1950 onward) for chosen countries from OurWorldInData.")
category = st.sidebar.selectbox("Select category", list(INDICATORS.keys()))
degree = st.sidebar.slider("Regression degree (min 3)", min_value=3, max_value=8, value=3)
year_increment = st.sidebar.selectbox("Graph display increments (years per plotted point on regression curve)", [1,2,5,10], index=0)
extrapolate_years = st.sidebar.number_input("Extrapolate into future (years)", min_value=0, max_value=100, value=10, step=1)
show_multiple = st.sidebar.checkbox("Show multiple countries on same graph (comparison)", value=True)
compare_us_groups = st.sidebar.checkbox("Compare Latin groups living in the United States (simplified)", value=False)
printer_friendly = st.sidebar.button("Generate printable report (opens download)")

# Utility: fetch data from OWID and tidy
@st.cache_data(ttl=3600)
def fetch_indicator_data(indicator_slug, countries, start_year=1950):
    url = get_owid_csv(indicator_slug, countries, start=start_year)
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.warning(f"Failed to fetch data for slug '{indicator_slug}' from OWID. URL attempted: {url}\\nError: {e}")
        return None
    # OWID returns columns: Entity, Year, Value
    # Pivot to have years as index and columns as countries
    if 'Year' in df.columns and 'Entity' in df.columns and 'Value' in df.columns:
        pivot = df.pivot(index='Year', columns='Entity', values='Value').sort_index()
        pivot.index = pivot.index.astype(int)
        return pivot
    else:
        # try to detect wide form already
        if 'year' in df.columns or 'Year' in df.columns:
            df.columns = [c if c!='year' else 'Year' for c in df.columns]
        return df

# Map category to slug (with fix)
slug = INDICATORS[category]["slug"]
slug = SLUG_FIXES.get(slug, slug)

if len(countries)==0:
    st.info("Please select at least one country.")
    st.stop()

# Fetch data
with st.spinner("Fetching historical data from OurWorldInData..."):
    data = fetch_indicator_data(slug, countries, start_year=1950)

if data is None or data.empty:
    st.error("No data available for the chosen category/countries combination. Try a different category or other countries.")
    st.stop()

# Trim to last 70 years from now (if requested)
current_year = datetime.now().year
start_year = max(min(data.index), current_year - 70 + 1)
data = data.loc[start_year:current_year]

# Show editable data table (use Streamlit data editor)
st.subheader("Raw data (editable)")
edited = st.data_editor(data.reset_index().rename(columns={"index":"Year"}), num_rows="dynamic")
# Convert back to years index
edited = edited.set_index("Year").sort_index()

# Prepare numeric arrays for regression: for each country, drop NaNs
def fit_and_analyze(years, values, deg=3, extrap_years=0):
    # polynomial fit in year as x
    x = np.array(years)
    y = np.array(values)
    # Fit on available points
    coeffs = np.polyfit(x, y, deg=deg)
    p = np.poly1d(coeffs)
    # Prepare evaluation grid: from min year to max year + extrap
    x_min, x_max = x.min(), x.max()
    x_eval = np.arange(x_min, x_max + extrap_years + 1)
    y_eval = p(x_eval)
    return {"poly":p, "coeffs":coeffs, "x_eval":x_eval, "y_eval":y_eval, "x":x, "y":y}

# Plotting
st.subheader("Scatter plot and regression curve")
fig = go.Figure()
colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]

analysis_results = {}
for i, country in enumerate(edited.columns):
    series = edited[country].dropna()
    if series.empty:
        st.warning(f"No data points for {country}; skipping.")
        continue
    years = series.index.astype(int).to_numpy()
    vals = series.to_numpy()
    res = fit_and_analyze(years, vals, deg=degree, extrap_years=extrapolate_years)
    analysis_results[country] = res
    # Scatter points
    fig.add_trace(go.Scatter(x=years, y=vals, mode='markers', name=f"{country} data", marker=dict(size=6)))
    # Regression curve: plot fitted curve up to last observed year
    x_fit = np.arange(years.min(), years.max()+1, year_increment)
    y_fit = res["poly"](x_fit)
    fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f"{country} fit (observed)", line=dict(width=3, color=colors[i%len(colors)])))
    # Extrapolated part (dashed)
    if extrapolate_years>0:
        x_extra = np.arange(years.max()+1, years.max()+extrapolate_years+1, year_increment)
        y_extra = res["poly"](x_extra)
        fig.add_trace(go.Scatter(x=x_extra, y=y_extra, mode='lines', name=f"{country} extrapolation", line=dict(width=2, dash='dash', color=colors[i%len(colors)])))
fig.update_layout(xaxis_title="Year", yaxis_title=f"{category} ({INDICATORS[category]['unit']})", height=600, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# Display equations and function analysis
st.subheader("Function (polynomial) equations & analysis")
for country, res in analysis_results.items():
    p = res["poly"]
    coeffs = res["coeffs"]
    # Equation string
    eq_terms = []
    deg = len(coeffs)-1
    for j,c in enumerate(coeffs):
        power = deg - j
        eq_terms.append(f"({c:.4g})*x^{power}")
    eq_str = " + ".join(eq_terms)
    st.markdown(f"**{country}** — fitted degree {deg}:  \nEquation: `{eq_str}`  ")
    # Derivative analysis
    dp = np.polyder(p)
    ddp = np.polyder(p, m=2)
    # Critical points: roots of derivative
    crits = np.roots(dp)
    real_crits = np.real(crits[np.isreal(crits)])
    # Evaluate only those within observed domain or within extrapolation if requested
    domain_min, domain_max = res["x"].min(), res["x"].max()
    crits_in_domain = [c for c in real_crits if domain_min <= c <= domain_max+extrapolate_years]
    # Determine maxima/minima by second derivative sign
    extrema_texts = []
    for c in crits_in_domain:
        sec = np.polyval(ddp, c)
        kind = "local minimum" if sec>0 else ("local maximum" if sec<0 else "inflection (flat)")
        val = p(c)
        # format as date-like string (year)
        year_str = f"{c:.2f}"
        extrema_texts.append(f"The {category.lower()} of **{country}** reached a {kind} on year {year_str}. The {category.lower()} was approximately {val:.2f} {INDICATORS[category]['unit']}.")
    if extrema_texts:
        for t in extrema_texts:
            st.write(t)
    else:
        st.write(f"No local extrema of the fitted polynomial for {country} lie inside the analyzed time window ({domain_min}–{domain_max+extrapolate_years}).")
    # Increasing/decreasing intervals (find sign changes of derivative)
    # Compute derivative roots and sort
    sorted_roots = sorted(real_crits)
    intervals = []
    test_points = []
    # construct intervals from domain_min to domain_max+extrapolate_years
    xs = [domain_min] + sorted_roots + [domain_max+extrapolate_years]
    for a,b in zip(xs[:-1], xs[1:]):
        mid = (a+b)/2.0
        slope = np.polyval(dp, mid)
        dir_text = "increasing" if slope>0 else ("decreasing" if slope<0 else "flat")
        intervals.append((a,b,dir_text,slope))
    st.write("Behavior over intervals (approx):")
    for a,b,dir_text,slope in intervals:
        st.write(f"From year {a:.2f} to {b:.2f}: **{dir_text}** (slope ≈ {slope:.4g} {INDICATORS[category]['unit']}/year)")
    # Fastest increase/decrease: point where second derivative zero? Actually maximum of first derivative => solve ddp=0 for real roots
    accel_roots = np.roots(ddp)
    real_accels = np.real(accel_roots[np.isreal(accel_roots)])
    best_points = []
    for r in real_accels:
        if domain_min <= r <= domain_max+extrapolate_years:
            rate = np.polyval(dp, r)
            best_points.append((r, rate))
    if best_points:
        # find largest positive (fastest increase) and most negative (fastest decrease)
        fastest_inc = max(best_points, key=lambda x: x[1])
        fastest_dec = min(best_points, key=lambda x: x[1])
        st.write(f"The {category.lower()} of **{country}** was increasing fastest at year {fastest_inc[0]:.2f} with instantaneous rate ≈ {fastest_inc[1]:.4g} {INDICATORS[category]['unit']}/year.")
        st.write(f"The {category.lower()} of **{country}** was decreasing fastest at year {fastest_dec[0]:.2f} with instantaneous rate ≈ {fastest_dec[1]:.4g} {INDICATORS[category]['unit']}/year.")
    else:
        st.write("No clear acceleration extrema inside domain to indicate fastest change based on the 2nd derivative root analysis.")

# Interpolation / Extrapolation tool
st.subheader("Interpolate / Extrapolate a value")
selected_country = st.selectbox("Choose country for value prediction", list(analysis_results.keys()))
input_year = st.number_input("Enter year to predict (can be outside observed range)", min_value=1900, max_value=2100, value=current_year+5)
if selected_country and input_year:
    poly = analysis_results[selected_country]["poly"]
    pred = float(poly(input_year))
    st.write(f"According to the regression model for **{selected_country}**, the {category.lower()} in the year {int(input_year)} is predicted to be **{pred:.3f} {INDICATORS[category]['unit']}** (extrapolation if outside observed data).")

# Average rate of change between two years (based on model)
st.subheader("Average rate of change between two years (model-based)")
country_for_rate = st.selectbox("Country for average rate", list(analysis_results.keys()), index=0)
year_a = st.number_input("Start year", min_value=1900, max_value=2100, value=int(start_year))
year_b = st.number_input("End year", min_value=1900, max_value=2100, value=int(current_year))
if country_for_rate and year_b>year_a:
    p = analysis_results[country_for_rate]["poly"]
    val_a = p(year_a)
    val_b = p(year_b)
    avg_rate = (val_b - val_a) / (year_b - year_a)
    st.write(f"Average rate of change for **{country_for_rate}** from {year_a} to {year_b} is **{avg_rate:.4g} {INDICATORS[category]['unit']}/year** (model-based).")
    st.write(f"Model predicts value {val_a:.3f} {INDICATORS[category]['unit']} in {year_a} and {val_b:.3f} {INDICATORS[category]['unit']} in {year_b}.")

# Printable report (generate simple HTML and make downloadable)
if printer_friendly:
    report_html = io.StringIO()
    report_html.write(f"<h1>Printable report — {category} — countries: {', '.join(analysis_results.keys())}</h1>")
    report_html.write(f"<h2>Generated on {datetime.now().isoformat()}</h2>")
    for country, res in analysis_results.items():
        report_html.write(f"<h3>{country}</h3>")
        report_html.write(f"<p>Equation (coefficients): {res['poly'].coeffs}</p>")
    st.download_button("Download printable report (HTML)", data=report_html.getvalue(), file_name="report.html", mime="text/html")

st.write("---")
st.caption("Notes: This app fetches historical indicator data from Our World in Data at runtime. OWID coverage and variable names vary by indicator and country. The polynomial regression is for demonstration and may not be the correct model for long-term forecasting. Use caution when extrapolating far into the future.")

