# first line: 115
@memory.cache
def get_cached_historical_data(asset_type, symbols, start_date, end_date):
    """
    Cached version of get_historical_data_for_asset_class for efficiency.
    """
    return get_historical_data_for_asset_class(asset_type, symbols, start_date, end_date)
