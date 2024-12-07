# first line: 87
@memory.cache
def get_cached_historical_data(asset_type, symbols, start_date, end_date):
    return get_historical_data_for_asset_class(asset_type, symbols, start_date, end_date)
