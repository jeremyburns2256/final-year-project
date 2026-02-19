"""
Basic BESS trading using a state machine.
No household consumption or generation.
No efficiency losses.
"""

import pandas as pd

# Setup cosntants
BESS_SIZE = 20 # kWH
BESS_INVERTER_CAPACITY = 5 # kW
BUY_THRESHOLD = 10 # $/MWH
SELL_THRESHOLD = 70 # $/MWH
BATTERY_INITIAL_STATE = 0

price_df = pd.read_csv('data/price_JAN26.csv')

BATTERY_STATE = BATTERY_INITIAL_STATE
BUY_COST = 0
SELL_REVENUE = 0

count_sell = 0
count_buy = 0
count_hold = 0

for index, row in price_df.iterrows():
    time = row['SETTLEMENTDATE']
    rrp = row['RRP']
    demand = row['TOTALDEMAND']

    if rrp < BUY_THRESHOLD and BATTERY_STATE < BESS_SIZE:
        BATTERY_STATE = min(BESS_SIZE, BATTERY_STATE + BESS_INVERTER_CAPACITY * (5/60)) # Charge at full inverter capacity for 5 minutes
        BUY_COST += BESS_INVERTER_CAPACITY * (5/60) * rrp / 1000 # Convert to MWH
        count_buy += 1

    elif rrp >= SELL_THRESHOLD and BATTERY_STATE > 0:
        BATTERY_STATE = max(0, BATTERY_STATE - BESS_INVERTER_CAPACITY * (5/60)) # Discharge at full inverter capacity for 5 minutes
        SELL_REVENUE += BESS_INVERTER_CAPACITY * (5/60) * rrp / 1000 # Convert to MWH
        count_sell += 1
    else:
        count_hold += 1


print(f"Final Battery State: {BATTERY_STATE:.2f} kWh")
print(f"Total Buy Cost: ${BUY_COST:.2f}")
print(f"Total Sell Revenue: ${SELL_REVENUE:.2f}")
print(f"Net Profit: ${SELL_REVENUE - BUY_COST:.2f}")

print(f"Buy Actions: {count_buy}")
print(f"Sell Actions: {count_sell}")
print(f"Hold Actions: {count_hold}")
