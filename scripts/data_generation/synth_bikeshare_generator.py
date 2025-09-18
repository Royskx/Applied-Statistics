import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Parameters
n_days = 365
hours = np.arange(24)
seasons = ["winter", "spring", "summer", "fall"]

rows = []
for day in range(n_days):
    season = seasons[(day // 90) % 4]
    is_weekend = (day % 7) in [5, 6]
    holiday = rng.random() < 0.05
    base_temp = {"winter": 5, "spring": 12, "summer": 24, "fall": 14}[season]
    for h in hours:
        temp = base_temp + 8*np.sin(2*np.pi*h/24) + rng.normal(0, 2)
        humidity = np.clip(60 + 20*np.sin(2*np.pi*(h-4)/24) + rng.normal(0, 10), 20, 100)
        wind = np.clip(abs(rng.normal(10, 5)), 0, 40)
        workingday = int((not is_weekend) and (not holiday))
        # demand mean
        mu = 50 + 10*workingday + 15*np.sin(2*np.pi*(h-8)/24) + 1.2*temp - 0.05*humidity - 0.3*wind
        mu = max(5, mu)
        rentals = rng.poisson(mu)
        rows.append({
            "day": day,
            "hour": int(h),
            "season": season,
            "holiday": int(holiday),
            "workingday": workingday,
            "temp": float(temp),
            "humidity": float(humidity),
            "windspeed": float(wind),
            "rentals": int(rentals),
        })

df = pd.DataFrame(rows)
df.to_csv("bikeshare_synth.csv", index=False)
print("Wrote bikeshare_synth.csv with", len(df), "rows")
