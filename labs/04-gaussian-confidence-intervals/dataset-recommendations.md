# Dataset Recommendations for Extended Confidence Interval Lab

## Purpose
Extend the Gaussian Confidence Intervals practical session with a more challenging real-world dataset that requires:
- Exploratory data analysis
- Distribution identification/fitting
- Parameter estimation using MLE or Method of Moments
- Approximate confidence intervals using CLT
- Interpretation in a practical context

---

## Recommended Datasets

### Option 1: NYC Taxi Trip Duration ⭐ **RECOMMENDED**

**Source:** Kaggle - NYC Taxi Trip Duration
**URL:** `https://www.kaggle.com/datasets/parisrohan/nyc-taxi-trip-duration`

**Description:**
- Real-world taxi trip data from NYC
- Contains pickup/dropoff locations, timestamps, and trip durations
- Large dataset (~1.5M rows) suitable for sampling exercises
- Trip duration is a continuous positive variable

**Variables:**
- `trip_duration` (seconds) - target variable
- `pickup_datetime`, `dropoff_datetime`
- `pickup_longitude`, `pickup_latitude`
- `dropoff_longitude`, `dropoff_latitude`
- `passenger_count`
- `store_and_fwd_flag`

**Statistical Challenges:**
1. **Distribution modeling**: Trip duration is right-skewed, potentially log-normal or gamma distributed
2. **Parameter estimation**: Estimate mean trip duration and variability
3. **CLT application**: Use CLT for confidence intervals despite non-normality
4. **Stratification**: Compare trip durations by time of day, day of week, or passenger count
5. **Practical interpretation**: What's the expected trip time? How reliable is our estimate?

**Pedagogical Value:**
- Students must justify distribution choice (visual + theoretical)
- Large sample size makes CLT approximation very accurate
- Natural subgroup analysis (rush hour vs off-peak)
- Real-world application with practical implications

**Suggested Exercises:**
1. Explore trip duration distribution (histogram, QQ-plots)
2. Test if log-transformation achieves normality
3. Estimate mean trip duration with 95% CI using CLT
4. Compare morning vs evening rush hour trip durations
5. Discuss practical implications for ride-hailing services

---

### Option 2: Bike Sharing Trip Duration

**Source:** Kaggle - NYC Citibike / Capital Bike Share
**URLs:**
- `https://www.kaggle.com/datasets/gabrielramos87/bike-trips`
- `https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset`

**Description:**
- Bike-sharing trip data from NYC or Washington DC
- Trip duration, start/end stations, user type
- Weather and seasonal data available in some versions

**Variables:**
- Trip duration (seconds/minutes)
- Start/end station
- User type (subscriber vs casual)
- Weather conditions (temp, humidity, windspeed)
- Date/time information

**Statistical Challenges:**
1. Duration distribution is heavily right-skewed (most trips short, some very long)
2. Potential mixture distribution (casual users vs commuters)
3. Seasonal effects on trip patterns
4. Outlier detection and handling (stolen bikes, forgot to return)

**Pedagogical Value:**
- Clear bimodal/mixture distribution teaching moment
- Outlier handling discussion
- Subgroup analysis by user type
- Real environmental/urban planning context

**Suggested Exercises:**
1. Compare trip duration distributions: subscribers vs casual users
2. Identify and justify removing outliers
3. Estimate mean duration for each user type with CIs
4. Test if weather affects trip duration (stratified analysis)
5. Discuss implications for bike availability planning

---

### Option 3: Airbnb Rental Prices

**Source:** Kaggle - New York Airbnb Open Data 2024
**URL:** `https://www.kaggle.com/datasets/vrindakallu/new-york-dataset`

**Description:**
- Airbnb listings with prices, locations, room types
- Moderate size dataset (~50k listings)
- Price is continuous positive variable

**Variables:**
- `price` (per night in USD)
- `room_type` (entire home, private room, shared room)
- `neighbourhood_group`, `neighbourhood`
- `latitude`, `longitude`
- `minimum_nights`
- `number_of_reviews`, `reviews_per_month`
- `availability_365`

**Statistical Challenges:**
1. Highly right-skewed price distribution
2. Strong effect of room type and location
3. Presence of outliers (luxury apartments)
4. Need for stratification or transformation

**Pedagogical Value:**
- Importance of stratification (by room type/borough)
- Log-transformation for interpretability
- Confidence intervals for median vs mean
- Real economic application

**Suggested Exercises:**
1. Explore price distribution by room type
2. Apply log-transformation and assess normality
3. Estimate average price for each room type with 95% CIs
4. Compare Manhattan vs other boroughs
5. Interpret results: what's a "typical" Airbnb price?

---

### Option 4: Wine Quality Dataset (UCI)

**Source:** UCI ML Repository / Kaggle
**URL:** `https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009`

**Description:**
- Physicochemical properties of Portuguese wines
- Quality scores from expert tasters
- Medium-sized dataset (~1600 samples)

**Variables:**
- `fixed acidity`, `volatile acidity`, `citric acid`
- `residual sugar`, `chlorides`
- `free sulfur dioxide`, `total sulfur dioxide`
- `density`, `pH`, `sulphates`, `alcohol`
- `quality` (score 0-10)

**Statistical Challenges:**
1. Multiple continuous variables to choose from
2. Quality score is discrete but can be treated as continuous
3. Natural to compare red vs white wines
4. Moderate sample size tests CLT assumptions

**Pedagogical Value:**
- Multiple variables allow different groups to work on different features
- Quality score has interesting bounded distribution
- Real sensory/chemistry application
- Can discuss sampling from production vs population

**Suggested Exercises:**
1. Choose a variable (e.g., alcohol content, pH)
2. Assess its distribution (visual + tests)
3. Estimate mean and variance with confidence intervals
4. Compare high vs low quality wines
5. Practical interpretation for winemakers

---

## Comparison Table

| Dataset | Size | Difficulty | CLT Relevance | Practical Context | Distribution Type |
|---------|------|------------|---------------|-------------------|-------------------|
| **NYC Taxi** | Large (1.5M) | Medium | Excellent | Transportation | Log-normal/Gamma |
| **Bike Share** | Large (100k+) | Medium-High | Excellent | Urban planning | Mixture/Exponential |
| **Airbnb** | Medium (50k) | Medium | Good | Economics | Log-normal |
| **Wine Quality** | Small (1.6k) | Low-Medium | Good | Chemistry | Various |

---

## Implementation Recommendation

### For the Lab Extension (Question 8+):

**Choice: NYC Taxi Trip Duration** ⭐

**Rationale:**
1. **Large sample size**: Makes CLT convergence very clear
2. **Real-world relevance**: Students understand the context immediately
3. **Natural log-transformation**: Log-normal is pedagogically useful
4. **Rich subgroup analysis**: Time of day, distance, etc.
5. **Available on Kaggle**: Easy download with API

**Proposed Structure:**

#### Question 8: Data Loading and Exploration
- Load a subset of NYC taxi data (e.g., 10,000 trips)
- Create histogram and summary statistics
- Identify that duration is right-skewed

#### Question 9: Distribution Assessment
- Plot QQ-plot against normal distribution
- Apply log-transformation
- Show log(duration) is approximately normal

#### Question 10: Parameter Estimation
- Estimate mean and variance of log(duration) using MLE
- Transform back to original scale
- Compute geometric mean of trip duration

#### Question 11: Confidence Interval Using CLT
- Construct 95% CI for mean log(duration) using CLT
- Transform CI back to original scale
- Interpret: "We are 95% confident that the geometric mean trip duration is between X and Y seconds"

#### Question 12: Subgroup Comparison
- Stratify by time period (e.g., 7-9am vs 7-9pm)
- Compute separate CIs for each group
- Determine if there's a significant difference

#### Question 13: Practical Interpretation
- Discuss implications for ride-hailing apps
- Why is the geometric mean more appropriate than arithmetic mean?
- How would you use this in practice?

---

## Data Access Instructions

### Method 1: Kaggle API (Recommended)
```python
# Install kaggle API
# !pip install kaggle

# Download dataset
import kaggle
kaggle.api.dataset_download_files('parisrohan/nyc-taxi-trip-duration',
                                   path='./data/',
                                   unzip=True)
```

### Method 2: Direct Download
Students can manually download from Kaggle (requires account)

### Method 3: Provide Preprocessed Subset
Create a curated subset (10k-50k rows) and host it in the course repository:
- Removes download complexity
- Ensures consistent results
- Faster computation
- Can preselect interesting time periods

---

## Alternative Simpler Option

If the taxi dataset is too complex, consider:

**Insurance Claims Duration** or **Customer Service Wait Times**
- Can generate synthetic data based on real patterns
- Exponential or Gamma distributed
- Smaller, more manageable
- Still pedagogically rich

---

## Pedagogical Goals

By the end of the extended lab, students should:
1. ✅ Select and justify an appropriate distribution for real data
2. ✅ Apply transformations when needed
3. ✅ Use CLT for non-normal data with large samples
4. ✅ Interpret confidence intervals in practical context
5. ✅ Compare groups using confidence intervals
6. ✅ Communicate statistical findings to non-technical audience

---

**Next Steps:**
1. Decide on dataset (recommend NYC Taxi)
2. Download and preprocess a subset
3. Create Questions 8-13 with solutions
4. Test with example analysis
5. Prepare dataset file for students

