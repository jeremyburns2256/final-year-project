"""
Module used for loading and normalising data
"""

import pandas as pd
from datetime import datetime, timedelta

def remove_outliers(price_df, column="RRP", lower_quantile=0.15, upper_quantile=0.85):
    lower_bound = price_df[column].quantile(lower_quantile)
    upper_bound = price_df[column].quantile(upper_quantile)
    return price_df[(price_df[column] >= lower_bound) & (price_df[column] <= upper_bound)]

def load_meter_data(csv_path, stream=None):
    """
    Load and parse NEM12 format meter data from CSV file.

    Args:
        csv_path: Path to the meter data CSV file in NEM12 format
        stream: Optional stream identifier (e.g., 'B1', 'E1', 'E6'). If None, returns all streams.

    Returns:
        pandas.DataFrame: DataFrame with 'timestamp' column and columns for each data stream.
                         Common streams: 'B1' (export = solar - load), 'E1' (grid import), 'E6' (controlled load)
    """
    data_streams = {}
    current_stream = None

    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')

            # 200 record type identifies the data stream
            if parts[0] == '200':
                current_stream = parts[3]  # Stream identifier (B1, E1, E6, etc.)
                if stream is None or current_stream == stream:
                    if current_stream not in data_streams:
                        data_streams[current_stream] = {'timestamps': [], 'values': []}

            # 300 record type contains interval data
            elif parts[0] == '300' and current_stream and (stream is None or current_stream == stream):
                if current_stream in data_streams:
                    date_str = parts[1]
                    date = datetime.strptime(date_str, '%Y%m%d')

                    # Extract 288 interval readings (5-minute intervals for 24 hours)
                    # Skip first 2 elements (record type and date)
                    intervals = parts[2:290]

                    for i, value in enumerate(intervals):
                        if value:  # Skip empty values
                            try:
                                reading = float(value)
                                # Calculate timestamp for this interval (5-minute intervals)
                                timestamp = date + timedelta(minutes=5*i)
                                data_streams[current_stream]['timestamps'].append(timestamp)
                                data_streams[current_stream]['values'].append(reading)
                            except ValueError:
                                continue

    # Build DataFrame from all streams
    if not data_streams:
        return pd.DataFrame(columns=['timestamp'])

    # Start with timestamps from the first stream
    first_stream = list(data_streams.keys())[0]
    df = pd.DataFrame({
        'timestamp': data_streams[first_stream]['timestamps'],
        first_stream: data_streams[first_stream]['values']
    })

    # Merge other streams if present
    for stream_name in list(data_streams.keys())[1:]:
        stream_df = pd.DataFrame({
            'timestamp': data_streams[stream_name]['timestamps'],
            stream_name: data_streams[stream_name]['values']
        })
        df = df.merge(stream_df, on='timestamp', how='outer')

    # Sort by timestamp and fill missing values
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

def export_meter_export_csv(meter_csv_path, output_csv_path):
    """
    Export grid export data from meter data.

    Converts B1 stream (export = solar - load, in kWh per 5min) to kW.

    NOTE: B1 represents NET EXPORT (solar - load), not solar generation.
    To get actual solar generation, you need separate solar metering.

    Args:
        meter_csv_path: Path to NEM12 meter data CSV
        output_csv_path: Path to write export CSV (SETTLEMENTDATE, EXPORT_KW format)
    """
    df = load_meter_data(meter_csv_path)

    # Convert kWh per 5min to kW: kWh / (5/60 hours) = kWh * 12
    export_df = pd.DataFrame({
        'SETTLEMENTDATE': df['timestamp'].dt.strftime('%-d/%m/%Y %-H:%M'),
        'EXPORT_KW': df['B1'] * 12  # Convert to kW
    })

    export_df.to_csv(output_csv_path, index=False)
    print(f"Exported {len(export_df)} export records to {output_csv_path}")

def export_meter_import_csv(meter_csv_path, output_csv_path):
    """
    Export grid import data from meter data.

    Converts E1 stream (grid import in kWh per 5min) to kW.

    Args:
        meter_csv_path: Path to NEM12 meter data CSV
        output_csv_path: Path to write import CSV (SETTLEMENTDATE, IMPORT_KW format)
    """
    df = load_meter_data(meter_csv_path)

    # Convert kWh per 5min to kW: kWh / (5/60 hours) = kWh * 12
    import_df = pd.DataFrame({
        'SETTLEMENTDATE': df['timestamp'].dt.strftime('%-d/%m/%Y %-H:%M'),
        'IMPORT_KW': df['E1'] * 12  # Convert to kW
    })

    import_df.to_csv(output_csv_path, index=False)
    print(f"Exported {len(import_df)} import records to {output_csv_path}")

def export_meter_load_csv(meter_csv_path, output_csv_path):
    """
    Export household load data from meter data to state machine format.

    WARNING: This function cannot accurately calculate total household load from meter
    data alone because B1 = export = (solar - load), not solar generation.

    To calculate total load, you need: load = solar - B1 (export)
    But solar is not available in the meter data - it requires separate solar metering.

    This function calculates GRID-SOURCED load only: E1 (import) + E6 (controlled load).
    This excludes any load served directly by solar.

    Converts from kWh per 5min to kW for state machine.

    Args:
        meter_csv_path: Path to NEM12 meter data CSV
        output_csv_path: Path to write load CSV (SETTLEMENTDATE, LOAD_KW format)
    """
    df = load_meter_data(meter_csv_path)

    # Grid-sourced load only = grid import + controlled load
    # NOTE: This does NOT include load served directly by solar
    # Total load = (solar - B1) + E1 + E6, but we don't have solar from meter data
    load_kwh_per_5min = df['E1'] + df['E6']

    # Convert kWh per 5min to kW: kWh / (5/60 hours) = kWh * 12
    load_df = pd.DataFrame({
        'SETTLEMENTDATE': df['timestamp'].dt.strftime('%-d/%m/%Y %-H:%M'),
        'LOAD_KW': load_kwh_per_5min * 12  # Convert to kW
    })

    load_df.to_csv(output_csv_path, index=False)
    print(f"Exported {len(load_df)} grid-sourced load records to {output_csv_path}")