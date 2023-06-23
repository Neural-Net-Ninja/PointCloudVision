from datetime import datetime

date_string = '20/06/2023'
date_format = '%d/%m/%Y'

# Convert the date string to a datetime object
dt = datetime.strptime(date_string, date_format)

# Set the time to 00:00:00
dt = dt.replace(hour=0, minute=0, second=0)

# Convert the datetime object to a Unix timestamp
timestamp = int(dt.timestamp())
print(timestamp)

